#!/bin/bash
# Hard-wipe every record in the brain `default` scenario.
#
# Why: smoke tests run via the ZeroClaw webhook, which reads/writes to
# the default scenario. Any leftover memories, tasks, entities,
# bookmarks, etc. get retrieved as `[Memory context]` for new prompts
# and can self-reinforce poison patterns from prior broken runs.
# `brain memory rm` only soft-deletes (sets is_deleted=true) but does
# not actually remove the records — semantic search may still pick
# them up. We hit SurrealDB directly with hard `DELETE`.
#
# Idempotent: zero records → still zero records, no error.
#
# Connection params come from
#   /home/r00t/workspace/projects/brain/llm-cli-rust/.env
# (OC_SURREAL_HOST/USER/PASS/NS/DB). The scenario record itself
# (`scenario:default`) is preserved so the wipe is record-deletion,
# not scenario-deletion.

set -u

ENV_FILE="${BRAIN_ENV_FILE:-/home/r00t/workspace/projects/brain/llm-cli-rust/.env}"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "[wipe-default] env file not found: $ENV_FILE" >&2
    exit 1
fi

# Pull connection params (only the OC_SURREAL_* keys).
SURREAL_HOST=$(grep -E "^OC_SURREAL_HOST=" "$ENV_FILE" | cut -d= -f2-)
SURREAL_USER=$(grep -E "^OC_SURREAL_USER=" "$ENV_FILE" | cut -d= -f2-)
SURREAL_PASS=$(grep -E "^OC_SURREAL_PASS=" "$ENV_FILE" | cut -d= -f2-)
SURREAL_NS=$(grep -E "^OC_SURREAL_NS=" "$ENV_FILE" | cut -d= -f2-)
SURREAL_DB=$(grep -E "^OC_SURREAL_DB=" "$ENV_FILE" | cut -d= -f2-)

if [[ -z "$SURREAL_HOST" || -z "$SURREAL_PASS" ]]; then
    echo "[wipe-default] missing OC_SURREAL_* env vars" >&2
    exit 1
fi

# Tables to wipe — every scenario-scoped table per
# brain-core/src/schema.rs (v112). `scenario:default` itself is
# preserved (it's a row in the `scenario` table; we never delete from
# `scenario`).
TABLES=(
    memory
    entity
    task
    bookmark
    project
    tool
    event
    location
    credential
    snippet
    routine
    media
    linked
    reminder
)

SQL=""
for t in "${TABLES[@]}"; do
    SQL+="DELETE $t WHERE scenario = 'default'; "
done

# Count records BEFORE wipe so we report what we deleted.
BEFORE=$(brain scenario list 2>/dev/null | awk '/^default/{print $4}')

# Pipe SQL into surreal CLI. Capture output for diagnostics.
OUT=$(echo "$SQL" | surreal sql \
    --endpoint "http://${SURREAL_HOST}" \
    --username "$SURREAL_USER" \
    --password "$SURREAL_PASS" \
    --namespace "$SURREAL_NS" \
    --database "$SURREAL_DB" 2>&1)
RC=$?

if [[ $RC -ne 0 ]]; then
    echo "[wipe-default] surreal sql FAILED (rc=$RC):" >&2
    echo "$OUT" | head -20 >&2
    exit $RC
fi

AFTER=$(brain scenario list 2>/dev/null | awk '/^default/{print $4}')
DELETED=$(( ${BEFORE:-0} - ${AFTER:-0} ))
echo "[wipe-default] before=${BEFORE:-?} after=${AFTER:-?} deleted=${DELETED} (memory/entity/task/bookmark/project/tool/event/location/credential/snippet/routine/media/linked/reminder)"
