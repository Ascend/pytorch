// Link-only marker for libraries for which no functions are directly imported.
extern "C" void fxrt_cann_stub_marker() {
  __builtin_trap();
}
