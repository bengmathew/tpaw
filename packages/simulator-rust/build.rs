use std::{
    fs::File,
    io::{Read, Result},
};
fn main() -> Result<()> {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=simulator-cuda");

    handle_proto_template("src/lib/wire/wire_common.proto.template");
    handle_proto_template("src/lib/wire/wire_simulate_api.proto.template");
    handle_proto_template("src/lib/wire/wire_estimate_portfolio_balance_api.proto.template");
    handle_proto_template("src/lib/wire/wire_plan_params_processed.proto.template");
    prost_build::compile_protos(
        &[
            "src/lib/wire/wire_common.proto",
            "src/lib/wire/wire_plan_params_server.proto",
            "src/lib/wire/wire_market_data_for_presets_processed.proto",
            "src/lib/wire/wire_simulate_api.proto",
            "src/lib/wire/wire_estimate_portfolio_balance_api.proto",
            "src/lib/wire/wire_plan_params_processed.proto",
        ],
        &["src/lib/wire/"],
    )?;
    Ok(())
}

fn handle_proto_template(path: &str) {
    // EFFICIENT MODE
    const FLOAT_WIRE: &str = "int64";

    // REPLICATION MODE
    // const FLOAT_WIRE: &str = "double";

    let mut contents = String::new();
    File::open(path)
        .unwrap()
        .read_to_string(&mut contents)
        .unwrap();

    let modified_contents = &contents;
    let modified_contents = &modified_contents.replace("{{float_wire}}", FLOAT_WIRE);
    let comment = "// Auto-generated by build.rs. Do not modify directly. Modify the .template file instead.\n\n";
    let modified_contents = comment.to_string() + &modified_contents;
    std::fs::write(path.replace(".template", ""), modified_contents).unwrap();
}
