// test feature derivec compilation errors
#[test]
fn feature_derive() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/feature/missing_dimension.rs");
    t.compile_fail("tests/feature/attribute.rs");
    t.compile_fail("tests/feature/wrong_dimension.rs");
    t.compile_fail("tests/feature/dimension_type.rs");
}
