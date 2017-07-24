

pub fn is_alphanumeric(str: &str) -> bool {
    for char in str.chars() {
        if !char.is_alphanumeric() {
            return false;
        }
    }
    true
}

pub fn is_control(str: &str) -> bool {
    for char in str.chars() {
        if char.is_control() {
            return true;
        }
    }
    false
}