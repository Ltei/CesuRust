
use network::cesure::Cesure;

pub struct CesureAndError {
    pub cesure : Cesure,
    pub error : f64,
}

impl CesureAndError {

    pub fn new(cesure : Cesure, error : f64) -> CesureAndError {
        return CesureAndError {
            cesure : cesure,
            error : error,
        }
    }

}