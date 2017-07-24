#![allow(dead_code)]


use std::f64;

use utils::matrix::Matrix;




pub fn mult_scl(matrix : &Matrix, val : f64) -> Matrix {
    let mut new_datas = Vec::with_capacity(matrix.len);
    for i in 0..matrix.len {
        new_datas.push(matrix.datas[i] * val);
    }
    return Matrix {
        rows : matrix.rows,
        cols : matrix.cols,
        len : matrix.len,
        datas : new_datas,
    };
}
pub fn div_scl(matrix : &Matrix, val : f64) -> Matrix {
    let mut new_datas = Vec::with_capacity(matrix.len);
    for i in 0..matrix.len {
        new_datas.push(matrix.datas[i] / val);
    }
    return Matrix {
        rows : matrix.rows,
        cols : matrix.cols,
        len : matrix.len,
        datas : new_datas,
    };
}

pub fn sub(matrix1 : &Matrix, matrix2 : &Matrix) -> Matrix {
    assert!(matrix1.rows == matrix2.rows && matrix1.cols == matrix2.cols);
    let mut new_datas = Vec::with_capacity(matrix1.len);
    for i in 0..matrix1.len {
        new_datas.push(matrix1.datas[i] - matrix2.datas[i]);
    }
    return Matrix {
        rows : matrix1.rows,
        cols : matrix1.cols,
        len : matrix1.len,
        datas : new_datas,
    };
}
pub fn p_mult(matrix1 : &Matrix, matrix2 : &Matrix) -> Matrix {
    if matrix1.rows != matrix2.rows || matrix1.cols != matrix2.cols {
        panic!("{} - {} {} - {}", matrix1.rows, matrix1.cols, matrix2.rows, matrix2.cols)
    }
    let mut new_datas = Vec::with_capacity(matrix1.len);
    for i in 0..matrix1.len {
        new_datas.push(matrix1.datas[i] * matrix2.datas[i]);
    }
    return Matrix {
        rows : matrix1.rows,
        cols : matrix1.cols,
        len : matrix1.len,
        datas : new_datas,
    };
}
/*pub fn m_dot(matrix1 : &Matrix, matrix2 : &Matrix) -> Matrix {
    let mut output = matrix1.clone();
    output.m_dot(matrix2);
    return output;
}*/
pub fn m_dot(matrix1 : &Matrix, matrix2 : &Matrix) -> Matrix {
    if matrix1.cols != matrix2.rows {
        panic!("matrix.m_dot - ({0} - {1}), ({2} - {3})", matrix1.rows, matrix1.cols, matrix2.rows, matrix2.cols);
    }
    let mut new_datas = vec![0.0; matrix1.rows*matrix2.cols];
    let mut new_datas_row_index = 0;
    let mut matrix1_datas_row_index = 0;
    for _ in 0..matrix1.rows {
        for col in 0..matrix2.cols {
            let mut data = &mut new_datas[new_datas_row_index+col];
            let mut matrix2_datas_index = col;
            for var in 0..matrix1.cols {
                *data += matrix1.datas[matrix1_datas_row_index+var] * matrix2.datas[matrix2_datas_index];
                matrix2_datas_index += matrix2.cols;
            }
        }
        new_datas_row_index += matrix2.cols;
        matrix1_datas_row_index += matrix1.cols;
    }
    return Matrix {
        rows : matrix1.rows,
        cols : matrix2.cols,
        len : matrix1.rows * matrix2.cols,
        datas : new_datas,
    }
}

pub fn transpose(matrix : &Matrix) -> Matrix {
    let new_rows = matrix.cols;
    let new_cols = matrix.rows;
    let mut new_datas = Vec::with_capacity(matrix.len);
    for col in 0..matrix.cols {
        for row in 0..matrix.rows {
            new_datas.push(matrix[(row,col)]);
        }
    }
    return Matrix {
        rows : new_rows,
        cols : new_cols,
        len : matrix.len,
        datas : new_datas,
    }
}

pub fn row_transpose(matrix : &Matrix) -> Matrix {
    assert!(matrix.is_row());
    return Matrix {
        rows : matrix.cols,
        cols : 1,
        len : matrix.len,
        datas : matrix.datas.clone(),
    }
}
pub fn row_append(matrix : &Matrix, val : f64) -> Matrix {
    assert!(matrix.is_row());
    let new_cols = matrix.cols+1;
    let new_len = matrix.len+1;
    let mut new_datas = Vec::with_capacity(new_len);
    for i in 0..matrix.len {
        new_datas.push(matrix.datas[i]);
    }
    new_datas.push(val);
    return Matrix {
        rows : matrix.rows,
        cols : new_cols,
        len : new_len,
        datas : new_datas,
    }
}
pub fn row_delete_last(matrix : &Matrix) -> Matrix {
    assert!(matrix.is_row() && matrix.len > 1);
    let new_cols = matrix.cols-1;
    let new_len = matrix.len-1;
    let mut new_datas = Vec::with_capacity(new_len);
    for i in 0..new_len {
        new_datas.push(matrix.datas[i]);
    }
    return Matrix {
        rows : matrix.rows,
        cols : new_cols,
        len : new_len,
        datas : new_datas,
    }
}
pub fn row_concatenate(matrix1 : &Matrix, matrix2 : &Matrix) -> Matrix {
    assert!(matrix1.is_row() && matrix2.is_row());
    let mut new_datas = Vec::with_capacity(matrix1.len+matrix2.len);
    for i in 0..matrix1.len {
        new_datas.push(matrix1.datas[i]);
    }
    for i in 0..matrix2.len {
        new_datas.push(matrix2.datas[i]);
    }
    return Matrix::new_row_from_datas(new_datas);
}
pub fn row_slice(matrix : &Matrix, left_len : usize) -> (Matrix, Matrix) {
    assert!(matrix.is_row() && left_len < matrix.len);
    let right_len = matrix.len-left_len;

    let mut datas1 = Vec::with_capacity(left_len);
    let mut datas2 = Vec::with_capacity(right_len);

    let mut index = 0;
    for _ in 0..left_len {
        datas1.push(matrix.datas[index]);
        index += 1;
    }
    for _ in 0..right_len {
        datas2.push(matrix.datas[index]);
        index += 1;
    }

    (Matrix::new_row_from_datas(datas1), Matrix::new_row_from_datas(datas2))
}
pub fn row_merge_avg(matrix1 : &Matrix, matrix2 : &Matrix) -> Matrix {
    assert!(matrix1.is_row() && matrix2.is_row() && matrix1.len == matrix2.len);
    let mut new_datas = Vec::with_capacity(matrix1.len);
    for i in 0..matrix1.len {
        new_datas.push( (matrix1.datas[i]+matrix2.datas[i]) / 2.0 );
    }
    return Matrix::new_row_from_datas(new_datas);
}


/*
#[allow(dead_code)]
pub fn vec_concatenate(matrix1 : &Matrix, matrix2 : &Matrix) -> Matrix {
    let mut new_datas = Vec::with_capacity(matrix1.len+matrix2.len);

    for i in 0..matrix1.len {
        new_datas.push(matrix1.datas[i]);
    }
    for i in 0..matrix2.len {
        new_datas.push(matrix2.datas[i]);
    }

    if matrix1.is_row() && matrix2.is_row() {
        return new_row_from_datas(new_datas);
    } else {
        assert!(matrix1.is_column() && matrix2.is_column());
        return new_col_from_datas(new_datas);
    }

    let mut output = matrix::new_row(matrix1.len+matrix2.len);
    let mut index = 0;
    for i in 0..matrix1.len {
        output.datas[index] = matrix1.datas[i];
        index += 1;
    }
    for i in 0..matrix2.len {
        output.datas[index] = matrix2.datas[i];
        index += 1;
    }
    return output;
}
*/

/*
#[allow(dead_code)]
pub fn vec_transpose(matrix : &Matrix) -> Matrix {
    let mut output = matrix.clone();
    output.vec_transpose();
    return output;
}
#[allow(dead_code)]
pub fn vec_append(vector : &Matrix, val : f64) -> Matrix {
    let mut output = vector.clone();
    output.vec_append(val);
    return output;
}
#[allow(dead_code)]
pub fn vec_delete_last(vector : &Matrix) -> Matrix {
    let mut output = vector.clone();
    output.vec_delete_last();
    return output;
}
*/