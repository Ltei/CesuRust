extern crate rand;
use rand::Rng;


pub struct Matrix {
    pub rows : usize,
    pub cols : usize,
    pub data : Vec<Vec<f64>>
}


#[allow(dead_code)]
pub fn new(rows :usize, cols :usize) -> Matrix {
    let matrix = Matrix {
        rows:rows,
        cols:cols,
        data:vec![vec![0.0; cols]; rows]
    };
    matrix
}

#[allow(dead_code)]
pub fn new_from_data(data :Vec<Vec<f64>>) -> Matrix {
    let rows : usize = data.len();
    let cols : usize = data[0].len();
    let mut matrix = Matrix {
        rows:rows,
        cols:cols,
        data:vec![vec![0.0; cols]; rows]
    };
    for row in 0..rows {
        for col in 0..cols {
            matrix.data[row][col] = data[row][col];
        }
    }
    return matrix;
}

#[allow(dead_code)]
pub fn new_row(len : usize) -> Matrix {
    return new(1,len);
}

#[allow(dead_code)]
pub fn new_row_from_data(data :Vec<f64>) -> Matrix {
    let mut matrix = new_row(data.len());
    for col in 0..matrix.cols {
        matrix.data[0][col] = data[col];
    }
    return matrix;
}

#[allow(dead_code)]
pub fn new_col(len : usize) -> Matrix {
    return new(len,1);
}

#[allow(dead_code)]
pub fn new_col_from_data(data :Vec<f64>) -> Matrix {
    let mut matrix = new_col(data.len());
    for row in 0..matrix.rows {
        matrix.data[row][0] = data[row];
    }
    return matrix;
}



impl Matrix {
    #[allow(dead_code)]
    pub fn print(&self) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let format = format!("{:.*}", 6, self.data[row][col]);
                print!("{:<9}", format);
                if col != self.cols-1 {
                    print!(" | ");
                }
            }
            println!();
        }
    }

    #[allow(dead_code)]
    pub fn print_title(&self, s :& str) {
        println!("{}",s);
        self.print();
    }


    #[allow(dead_code)]
    pub fn copy(&self) -> Matrix {
        let mut matrix = Matrix {
            rows:self.rows,
            cols:self.cols,
            data:vec![vec![0.0; self.cols]; self.rows]
        };
        for row in 0..self.rows {
            for col in 0..self.cols {
                matrix.data[row][col] = self.data[row][col];
            }
        }
        return matrix;
    }


    #[allow(dead_code)]
    pub fn is_row(&self) -> bool {
        return self.rows == 1;
    }

    #[allow(dead_code)]
    pub fn is_column(&self) -> bool {
        return self.cols == 1;
    }

    #[allow(dead_code)]
    pub fn is_vector(&self) -> bool {
        return self.rows == 1 || self.cols == 1;
    }


    #[allow(dead_code)]
    pub fn get_row(&self, row:usize) -> Matrix {
        if row >= self.rows {
            panic!("matrix.get_row - {0}, {1}", self.rows, row);
        }
        let mut matrix = new_row(self.cols);
        for col in 0..self.cols {
            matrix.data[row][col] = self.data[row][col];
        }
        return matrix;
    }

    #[allow(dead_code)]
    pub fn get_col(&self, col:usize) -> Matrix {
        if col >= self.cols {
            panic!("matrix.get_col - {0}, {1}", self.cols, col);
        }
        let mut matrix = new_col(self.rows);
        for row in 0..self.rows {
            matrix.data[row][col] = self.data[row][col];
        }
        return matrix;
    }

    #[allow(dead_code)]
    pub fn avg(&self) -> f64 {
        let mut avg = 0.0;
        for row in 0..self.rows {
            for col in 0..self.cols {
                avg += self.data[row][col];
            }
        }
        avg /= (self.rows * self.cols) as f64;
        return avg;
    }


    #[allow(dead_code)]
    pub fn set_zero(&mut self) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] = 0.;
            }
        }
    }

    #[allow(dead_code)]
    pub fn set_random(&mut self, min :f64, max :f64, mut rand : & mut rand::ThreadRng) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] = rand.gen_range::<f64>(min,max);
            }
        }
    }

    #[allow(dead_code)]
    pub fn set_random_int(&mut self, min :i32, max :i32, mut rand : rand::ThreadRng) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] = rand.gen_range::<i32>(min, max+1) as f64;
            }
        }
    }



    #[allow(dead_code)]
    pub fn transpose<'a>(&'a mut self) -> &'a Matrix {
        let mut new_data = vec![vec![0.0; self.rows]; self.cols];
        for row in 0..self.rows {
            for col in 0..self.cols {
                new_data[col][row] = self.data[row][col];
            }
        }
        self.data = new_data;
        return self;
    }

    #[allow(dead_code)]
    pub fn add<'a>(&'a mut self, matrix : Matrix) -> &'a Matrix {
        if self.rows != matrix.rows || self.cols != matrix.cols {
            panic!("matrix.add - {0}, {1}, {2}, {3}", self.rows, matrix.rows, self.cols, matrix.cols);
        }
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] += matrix.data[row][col];
            }
        }
        return self;
    }

    #[allow(dead_code)]
    pub fn sub<'a>(&'a mut self, matrix : Matrix) -> &'a Matrix {
        if self.rows != matrix.rows || self.cols != matrix.cols {
            panic!("matrix.sub - {0}, {1}, {2}, {3}", self.rows, matrix.rows, self.cols, matrix.cols);
        }
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] -= matrix.data[row][col];
            }
        }
        return self;
    }

    #[allow(dead_code)]
    #[inline]
    pub fn p_mult(mut self, matrix : & Matrix) -> Matrix {
        if self.rows != matrix.rows || self.cols != matrix.cols {
            panic!("matrix.p_mult - {0}, {1}, {2}, {3}", self.rows, matrix.rows, self.cols, matrix.cols);
        }
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] *= matrix.data[row][col];
            }
        }
        return self;
    }

    #[allow(dead_code)]
    #[inline]
    pub fn m_dot(mut self, matrix : & Matrix) -> Matrix {
        if self.cols != matrix.rows {
            panic!("matrix.m_dot - {0} - {1}, {2} - {3}", self.cols, self.rows, matrix.cols, matrix.rows);
        }
        let mut new_data = vec![vec![0.0; matrix.cols]; self.rows];
        for row in 0..self.rows {
            for col in 0..matrix.cols {
                for var in 0..self.cols {
                    new_data[row][col] += self.data[row][var] * matrix.data[var][col];
                }
            }
        }
        self.data = new_data;
        return self;
    }


    #[allow(dead_code)]
    pub fn mult_scl(mut self, scalar : f64) -> Matrix {
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] *= scalar;
            }
        }
        return self;
    }

    #[allow(dead_code)]
    pub fn div_scl(mut self, scalar : f64) -> Matrix {
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] /= scalar;
            }
        }
        return self;
    }


    #[allow(dead_code)]
    pub fn del_last_col(mut self) -> Matrix {
        let mut new_data = vec![vec![0.0; self.rows]; self.cols-1];
        for row in 0..self.rows {
            for col in 0..self.cols-1 {
                    new_data[row][col] = self.data[row][col];
            }
        }
        self.data = new_data;
        return self;
    }
}