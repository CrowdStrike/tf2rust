import utils.model_saver as model_saver
import utils.rust_converter as rust_converter

if __name__ == "__main__":
    model_saver.save_tf_model()
    rust_converter.convert_to_rust()
