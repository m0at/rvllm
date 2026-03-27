#!/bin/bash
INSTANCE_ID=${1:-$(cat deploy/.instance_id 2>/dev/null)}
echo "Destroying instance $INSTANCE_ID..."
vastai destroy instance $INSTANCE_ID
rm -f deploy/.instance_id
echo "Done."
