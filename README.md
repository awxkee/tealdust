# tealdust

Still images AV2 AVIFs decoder in Rust.

## Example

```rust
use tealdust::AvifDecoder;

fn main() {
    let bytes = std::fs::read("./assets/image.avif").unwrap();

    let mut decoder = AvifDecoder::new(&bytes).unwrap();
    let image = decoder.decode().unwrap();

    println!(
        "{}x{} {:?}, {} bpc",
        image.width, image.height, image.pixel_layout, image.bits_per_component,
    );

    // Decoded Y/U/V plane bytes (row-major), with per-plane row strides.
    let _luma = &image.planes[0];
}
```

## License

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.