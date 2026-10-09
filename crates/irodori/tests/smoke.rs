use irodori::{Device, Tensor};

#[test]
fn cpu_matmul_works() {
    let dev = Device::flex();
    let a = Tensor::<2>::from_floats([[1., 2.], [3., 4.]], &dev);
    let b = Tensor::<2>::from_floats([[1., 0.], [0., 1.]], &dev);
    let c = a.matmul(b).into_data().to_vec::<f32>().unwrap();
    assert_eq!(c, vec![1., 2., 3., 4.]);
}
