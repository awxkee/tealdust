use tealdust::{AvifDecoder, AvifSettings};

fn main() {
    let path = std::env::args().nth(1).expect("AVIF path");
    let bytes = std::fs::read(path).unwrap();
    let mut settings = AvifSettings::default();
    settings.decoder_settings.frame_size_limit = 0;
    match AvifDecoder::with_settings(&bytes, settings) {
        Ok(mut decoder) => {
            eprintln!("info: {:?}", decoder.image_info());
            match decoder.decode() {
                Ok(image) => eprintln!(
                    "decoded: {}x{} {:?} alpha={}",
                    image.width,
                    image.height,
                    image.pixel_layout,
                    image.alpha.is_some()
                ),
                Err(error) => eprintln!("decode error: {error:?}"),
            }
        }
        Err(error) => eprintln!("open error: {error:?}"),
    }
}
