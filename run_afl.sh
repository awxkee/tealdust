cargo afl build --release --package tealdust-afl

AFL_SKIP_CPUFREQ=1 \
AFL_FAST_CAL=1 \
AFL_NO_STARTUP_CALIBRATION=1 \
AFL_CMPLOG_ONLY_NEW=1 \
cargo afl fuzz \
  -i /Users/radzivon/Downloads/avif_corpus \
  -o ./findings \
  -t 1000+ \
  -m none \
  -c - \
  -- ./target/release/tealdust-afl