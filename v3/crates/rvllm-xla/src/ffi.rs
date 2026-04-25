use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PjrtElementType {
    S8,
    S32,
    U8,
    F32,
    BF16,
}
