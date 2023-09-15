from .utils import model_saver, rust_converter

if __name__ == "__main__":
    model_saver.save_tf_model()
    rust_converter.convert_to_rust()
