use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(Feature, attributes(dimension))]
pub fn derive_feature(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_feature_impl(input) {
        Ok(token_stream) => token_stream,
        Err(e) => e.to_compile_error().into(),
    }
}

fn derive_feature_impl(input: DeriveInput) -> syn::Result<proc_macro::TokenStream> {
    let name = input.ident;

    let mut dimension: Option<usize> = None;

    for attr in input.attrs {
        if attr.path().is_ident("dimension") {
            if let syn::Meta::NameValue(meta) = attr.meta {
                if let syn::Expr::Lit(val) = meta.value {
                    if let syn::Lit::Int(v) = val.lit {
                        dimension = Some(v.base10_parse::<usize>().unwrap());
                    }
                }
            }
        }
    }

    let dim = dimension.ok_or_else(|| {
        syn::Error::new_spanned(
            name.clone(),
            "Missing #[dimension = <d>] attribute where d of type usize is equal to input dimension of feature function",
        )
    })?;

    let expanded = quote! {
        impl Feature<#dim> for #name {
            fn val(&self, x: &nalgebra::SVector<f64, #dim>) -> f64 {
                self.f(&x)
            }

            fn val_grad(&self, x: &nalgebra::SVector<f64, #dim>) -> (f64, nalgebra::SVector<f64,#dim>) {
                num_dual::gradient(|v| self.f(&v), *x)
            }

            fn val_grad_hes(&self, x: &nalgebra::SVector<f64, #dim>) -> (f64, nalgebra::SVector<f64, #dim>, nalgebra::SMatrix<f64, #dim, #dim>) {
                num_dual::hessian(|v| self.f(&v), *x)
            }
        }
    };
    Ok(proc_macro::TokenStream::from(expanded))
}
