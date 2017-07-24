#![allow(dead_code)]

extern crate rand;

use rand::Rng;

use std::clone::Clone;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Add;

use utils::math;
use utils::traits::Parse;



pub struct Matrix {
    pub rows : usize,
    pub cols : usize,
    pub len : usize,
    pub datas : Vec<f64>
}


impl Clone for Matrix {
    fn clone(&self) -> Matrix {
        return Matrix {
            rows : self.rows,
            cols : self.cols,
            len : self.len,
            datas : self.datas.clone()
        }
    }
    fn clone_from(&mut self, source: &Matrix) {
        self.rows = source.rows;
        self.cols = source.cols;
        self.len = source.len;
        self.datas.clone_from(&source.datas);
    }
}
impl Index<(usize,usize)> for Matrix {
    type Output = f64;
    fn index(&self, index:(usize,usize)) -> &f64 {
        return &self.datas[index.0*self.cols + index.1]
    }
}
impl IndexMut<(usize,usize)> for Matrix {
    fn index_mut(&mut self, index:(usize,usize)) -> &mut f64 {
        return &mut self.datas[index.0*self.cols + index.1]
    }
}
impl Parse for Matrix {
    fn to_string(&self) -> String {
        let mut output = format!("{} {}", self.rows, self.cols);
        for i in 0..self.len {
            output = output.add(&format!(" {}", self.datas[i]));
        }
        return output;
    }
    fn from_string(str : &str) -> Matrix {
        let parsed : Vec<&str> = str.split(" ").collect();
        let rows : usize = parsed[0].parse().unwrap();
        let cols : usize = parsed[1].parse().unwrap();
        let len = rows * cols;
        let mut datas : Vec<f64> = Vec::with_capacity(len);
        for i in 2..parsed.len() {
            datas.push(parsed[i].parse().unwrap());
        }
        assert!(datas.len() == len);
        return Matrix {
            rows : rows,
            cols : cols,
            len : len,
            datas : datas,
        }
    }
}



impl Matrix {

    #[inline]
    pub fn new(rows : usize, cols : usize) -> Matrix {
        assert!(rows > 0 && cols > 0);
        return Matrix {
            rows : rows,
            cols : cols,
            len : rows*cols,
            datas : vec![0.0; rows*cols]
        };
    }
    #[inline]
    pub fn new_from_datas(datas :Vec<Vec<f64>>) -> Matrix {
        let rows : usize = datas.len();
        let cols : usize = datas[0].len();
        assert!(rows > 0 && cols > 0);
        let len : usize = rows * cols;

        let mut new_datas = Vec::with_capacity(len);
        for row in 0..rows {
            assert!(datas[row].len() == cols);
            for col in 0..cols {
                new_datas.push(datas[row][col]);
            }
        }

        return Matrix {
            rows : rows,
            cols : cols,
            len : len,
            datas : new_datas,
        };
    }
    #[inline]
    pub fn new_row(len : usize) -> Matrix {
        return Matrix::new(1,len);
    }
    #[inline]
    pub fn new_row_from_datas(datas :Vec<f64>) -> Matrix {
        let rows : usize = 1;
        let cols : usize = datas.len();
        return Matrix {
            rows : rows,
            cols : cols,
            len : rows*cols,
            datas : datas,
        };
    }
    #[inline]
    pub fn new_col(len : usize) -> Matrix {
        return Matrix::new(len,1);
    }
    #[inline]
    pub fn new_col_from_datas(datas :Vec<f64>) -> Matrix {
        let rows : usize = datas.len();
        let cols : usize = 1;
        return Matrix {
            rows : rows,
            cols : cols,
            len : rows*cols,
            datas : datas,
        };
    }

    #[inline]
    pub fn clone_randomized(&self, magnitude : f64, rand : &mut rand::ThreadRng) -> Matrix {
        let mut new_datas = Vec::with_capacity(self.len);
        for i in 0..self.len {
            new_datas.push(self.datas[i] + math::rand(-magnitude, magnitude, rand));
        }
        return Matrix {
            rows : self.rows,
            cols : self.cols,
            len : self.len,
            datas : new_datas,
        }
    }

    #[inline]
    pub fn print(&self) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let format = format!("{:.*}", 6, self[(row,col)]);
                print!("{:<9}", format);
                if col != self.cols-1 {
                    print!(" | ");
                }
            }
            println!();
        }
    }
    #[inline]
    pub fn print_title(&self, s :& str) {
        println!("{}",s);
        self.print();
    }

    #[inline]
    pub fn set_zero(&mut self) -> &mut Matrix {
        for index in 0..self.len {
            self.datas[index] = 0.0;
        }
        return self;
    }
    #[inline]
    pub fn set_random(&mut self, min :f64, max :f64, mut rand : &mut rand::ThreadRng) {
        for index in 0..self.len {
            self.datas[index] = rand.gen_range::<f64>(min,max);
        }
    }
    #[inline]
    pub fn set_random_int(&mut self, min :i32, max :i32, mut rand : &mut rand::ThreadRng) {
        for index in 0..self.len {
            self.datas[index] = rand.gen_range::<i32>(min, max+1) as f64;
        }
    }

    #[inline]
    pub fn is_row(&self) -> bool {
        return self.rows == 1;
    }
    #[inline]
    pub fn is_column(&self) -> bool {
        return self.cols == 1;
    }
    #[inline]
    pub fn is_vector(&self) -> bool {
        return self.rows == 1 || self.cols == 1;
    }
    #[inline]
    pub fn is_finite(&self) -> bool {
        for i in 0..self.len {
            if !self.datas[i].is_finite() {
                return false;
            }
        }
        return true;
    }
    #[inline]
    pub fn is_always_less_than(&self, val : f64) -> bool {
        for i in 0..self.len {
            if self.datas[i] >= val {
                return false;
            }
        }
        return true;
    }

    #[inline]
    pub fn delete_last_col(&mut self) -> &mut Matrix {
        assert!(self.cols > 1);
        let new_len = self.len - self.rows;
        let new_cols = self.cols - 1;
        let mut new_datas = vec![0.0; new_len];
        for row in 0..self.rows {
            for col in 0..new_cols {
                new_datas[row*new_cols+col] = self[(row,col)];
            }
        }
        self.len = new_len;
        self.cols = new_cols;
        self.datas = new_datas;
        return self;
    }

    #[inline]
    pub fn add(&mut self, matrix : &Matrix) -> &mut Matrix {
        if self.rows != matrix.rows || self.cols != matrix.cols {
            panic!("matrix.add - ({0} - {1}), ({2} - {3})", self.rows, self.cols, matrix.rows, matrix.cols);
        }
        for index in 0..self.len {
            self.datas[index] += matrix.datas[index];
        }
        return self;
    }
    #[inline]
    pub fn sub(&mut self, matrix : &Matrix) -> &mut Matrix {
        if self.rows != matrix.rows || self.cols != matrix.cols {
            panic!("matrix.sub - ({0} - {1}), ({2} - {3})", self.rows, self.cols, matrix.rows, matrix.cols);
        }
        for index in 0..self.len {
            self.datas[index] -= matrix.datas[index];
        }
        return self;
    }
    #[inline]
    pub fn p_mult(&mut self, matrix : &Matrix) -> &mut Matrix {
        if self.rows != matrix.rows || self.cols != matrix.cols {
            panic!("matrix.p_mult - ({0} - {1}), ({2} - {3})", self.rows, self.cols, matrix.rows, matrix.cols);
        }
        for index in 0..self.len {
            self.datas[index] *= matrix.datas[index];
        }
        return self;
    }
    #[inline]
    pub fn m_dot(&mut self, matrix : &Matrix) -> &mut Matrix {
        if self.cols != matrix.rows {
            panic!("matrix.m_dot - ({0} - {1}), ({2} - {3})", self.rows, self.cols, matrix.rows, matrix.cols);
        }
        let mut new_datas = vec![0.0; self.rows*matrix.cols];
        {
            for row in 0..self.rows {
                for col in 0..matrix.cols {
                    let mut data = &mut new_datas[row*matrix.cols+col];
                    for var in 0..self.cols {
                        *data += self.datas[row*self.cols+var] * matrix.datas[var*matrix.cols+col];
                    }
                }
            }
        }
        self.datas = new_datas;
        self.cols = matrix.cols;
        self.len = self.rows * self.cols;
        return self;
    }
    #[inline]
    pub fn mult_scl(&mut self, val : f64) -> &mut Matrix {
        for index in 0..self.len {
            self.datas[index] *= val;
        }
        return self;
    }
    #[inline]
    pub fn div_scl(&mut self, val : f64) -> &mut Matrix {
        for index in 0..self.len {
            self.datas[index] /= val;
        }
        return self;
    }
    #[inline]
    pub fn abs(&mut self) -> &mut Matrix {
        for i in 0..self.len {
            self.datas[i] = math::abs(self.datas[i]);
        }
        return self;
    }

    #[inline]
    pub fn get_row(&self, row:usize) -> Matrix {
        if row >= self.rows {
            panic!("matrix.get_row - {0}, {1}", self.rows, row);
        }
        let mut row_datas = Vec::with_capacity(self.cols);
        for col in 0..self.cols {
            row_datas.push(self[(row,col)]);
        }
        return Matrix::new_row_from_datas(row_datas);
    }
    #[inline]
    pub fn get_col(&self, col:usize) -> Matrix {
        if col >= self.cols {
            panic!("matrix.get_col - {0}, {1}", self.cols, col);
        }
        let mut col_datas = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            col_datas.push(self[(row,col)]);
        }
        return Matrix::new_col_from_datas(col_datas);
    }
    #[inline]
    pub fn get_avg(&self) -> f64 {
        let mut avg = 0.0;
        for index in 0..self.len {
            avg += self.datas[index];
        }
        avg /= (self.rows * self.cols) as f64;
        return avg;
    }
    #[inline]
    pub fn get_abs_avg(&self) -> f64 {
        let mut avg = 0.0;
        for index in 0..self.len {
            avg += math::abs(self.datas[index]);
        }
        avg /= (self.rows * self.cols) as f64;
        return avg;
    }

    #[inline]
    pub fn row_append(&mut self, val : f64) -> &mut Matrix {
        assert!(self.is_row());
        self.cols += 1;
        self.len += 1;
        self.datas.push(val);
        return self;
    }
    #[inline]
    pub fn row_delete_last(&mut self) -> &mut Matrix {
        assert!(self.is_row() && self.len > 1);
        self.cols -= 1;
        self.len -= 1;
        self.datas.pop();
        return self;
    }
    #[inline]
    pub fn row_concatenate(&mut self, matrix : &Matrix) -> &mut Matrix {
        assert!(self.is_row() && matrix.is_row());
        self.cols += matrix.cols;
        self.len += matrix.len;
        for i in 0..matrix.len {
            self.datas.push(matrix.datas[i]);
        }
        self
    }

}



/*#[inline]
    pub fn transpose(&mut self) -> &mut Matrix {
        let mut new_datas = vec![0.0; self.len];
        for row in 0..self.rows {
            for col in 0..self.cols {
                new_datas[col*self.rows+row] = self[(row,col)];
            }
        }
        self.datas = new_datas;
        return self;
    }*/
/*#[inline]
    pub fn m_dot(&mut self, matrix : &Matrix) -> &mut Matrix {
        if self.cols != matrix.rows {
            panic!("matrix.m_dot - {0} - {1}, {2} - {3}", self.cols, self.rows, matrix.cols, matrix.rows);
        }
        let mut new_datas = vec![0.0; self.rows*matrix.cols];
        {
            let self_datas = &self.datas;
            let matrix_datas = &matrix.datas;
            let mut new_datas_index = 0;
            let mut self_datas_row_index = 0;
            for _ in 0..self.rows {
                for col in 0..matrix.cols {
                    let mut data = &mut new_datas[new_datas_index];
                    *data = 0.0;
                    let mut matrix_datas_index = col;
                    for var in 0..self.cols {
                        println!("{} - {}", self_datas_row_index+var, matrix_datas_index);
                        *data += self_datas[self_datas_row_index+var] * matrix_datas[matrix_datas_index];
                        matrix_datas_index += matrix.cols;
                    }
                    new_datas_index += 1;
                }
                self_datas_row_index += self.rows;
            }
        }
        self.datas = new_datas;
        self.cols = matrix.cols;
        self.len = self.rows * self.cols;
        return self;
    }*/
/*#[inline]
    pub fn vec_transpose(&mut self) -> &mut Matrix {
        assert!(self.is_vector());
        let rows = self.rows;
        self.rows = self.cols;
        self.cols = rows;
        return self;
    }
    #[inline]
    pub fn vec_append(&mut self, val : f64) -> &mut Matrix {
        if self.is_row() {
            self.cols += 1;
        } else {
            assert!(self.is_column());
            self.rows += 1;
        }
        self.len += 1;
        self.datas.push(val);
        return self;
    }
    #[inline]
    pub fn vec_delete_last(&mut self) -> &mut Matrix {
        if self.is_row() {
            self.cols -= 1;
        } else {
            assert!(self.is_column());
            self.rows -= 1;
        }
        self.len -= 1;
        self.datas.pop();
        return self;
    }*/