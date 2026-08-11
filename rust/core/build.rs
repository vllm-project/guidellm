use vergen_gix::{Emitter, Gix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    Emitter::default()
        .add_instructions(&Gix::all_git())?
        .emit()?;
    Ok(())
}
