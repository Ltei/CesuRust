

#[allow(dead_code)]
impl Matrix {

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
    pub fn get_row(&self, row:usize) -> Matrix {
        if row >= self.rows {
            panic!("matrix.get_row - {0}, {1}", self.rows, row);
        }
        let mut row_datas = Vec::with_capacity(self.cols);
        for col in 0..self.cols {
            row_datas.push(self[(row,col)]);
        }
        return new_row_from_datas(row_datas);
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
        return new_col_from_datas(col_datas);
    }


    #[inline]
    pub fn avg(&self) -> f64 {
        let mut avg = 0.0;
        for index in 0..self.len {
            avg += self.datas[index];
        }
        avg /= (self.rows * self.cols) as f64;
        return avg;
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
    pub fn transpose(&mut self) -> &mut Matrix {
        let mut new_datas = vec![0.0; self.len];
        for row in 0..self.rows {
            for col in 0..self.cols {
                new_datas[col*self.rows+row] = self[(row,col)];
            }
        }
        self.datas = new_datas;
        return self;
    }
    #[inline]
    pub fn delete_last_col(&mut self) -> &mut Matrix {
        if self.cols <= 1 {
            panic!("matrix.delete_last_col - {0}, {1}", self.rows, self.cols);
        }
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
            panic!("matrix.add - {0}, {1}, {2}, {3}", self.rows, matrix.rows, self.cols, matrix.cols);
        }
        for index in 0..self.len {
            self.datas[index] += matrix.datas[index];
        }
        return self;
    }
    #[inline]
    pub fn sub(&mut self, matrix : &Matrix) -> &mut Matrix {
        if self.rows != matrix.rows || self.cols != matrix.cols {
            panic!("matrix.sub - {0}, {1}, {2}, {3}", self.rows, matrix.rows, self.cols, matrix.cols);
        }
        for index in 0..self.len {
            self.datas[index] -= matrix.datas[index];
        }
        return self;
    }
    #[inline]
    pub fn p_mult(&mut self, matrix : &Matrix) -> &mut Matrix {
        if self.rows != matrix.rows || self.cols != matrix.cols {
            panic!("matrix.p_mult - {0}, {1}, {2}, {3}", self.rows, matrix.rows, self.cols, matrix.cols);
        }
        for index in 0..self.len {
            self.datas[index] *= matrix.datas[index];
        }
        return self;
    }
    #[inline]
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
                        *data += self_datas[self_datas_row_index+var] * matrix_datas[matrix_datas_index];
                        matrix_datas_index += matrix.rows;
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

}