
extern crate rand;

use rand::Rng;


#[inline]
pub fn rand(min: f64, max: f64, rand: &mut rand::ThreadRng) -> f64 {
    match min == max {
        true => { return min; },
        false => {
            match min > max {
                true => { return rand.gen_range::<f64>(max, min); },
                false => { return rand.gen_range::<f64>(min, max); }, }
        }
    }
}


#[inline]
pub fn abs(x : f64) -> f64 {
    if x < 0.0 {
        return -x;
    } else {
        return x;
    }
}
#[inline]
pub fn square(x : f64) -> f64 {
    return x*x;
}



#[inline]
pub fn sigmoid(x : f64) -> f64 {
    return 0.5 - 0.5 * (x / (1.0 + abs(x)));
}
#[inline]
pub fn sigmoid_deriv(x : f64) -> f64 {
    return -0.5/square(1.0 + abs(x));
}
#[inline]
pub fn tanh(x : f64) -> f64 {
    return x/(1.0+abs(x));
}
#[inline]
pub fn tanh_deriv(x : f64) -> f64 {
    return 1.0/square(1.0+abs(x));
}