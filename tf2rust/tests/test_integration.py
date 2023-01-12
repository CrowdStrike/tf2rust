import collections
import filecmp
import os
import pathlib
import shutil
import subprocess
import unittest

collections.Callable = collections.abc.Callable

# Models to test the TensorFlow-to-Rust conversion for.
MODELS = ["sample_v1"]

TESTS_PATH = pathlib.Path(__file__).parent.absolute()

# Path where we need to have the TensorFlow model (ft_model), but also a numpy zip containing
# features and TensorFlow predictions for those features, in order to check that Rust predictions
# match TensorFlow predictions.
MODELS_METADATA = TESTS_PATH / "data"

# Path where we'll store the Rust generated code that results from the TensorFlow-to-Rust
# conversion.
RUST_GENERATED_CODE = TESTS_PATH / "generated_code"

# Path where we'll store the Rust expected code that should be generated.
RUST_GENERATED_CODE_EXPECTED = TESTS_PATH / "generated_code_expected"


class ModelConversionTest(unittest.TestCase):
    def test_model_conversion(self):
        """
        Tests model conversion for all the provided models.
        """
        for model in MODELS:
            with self.subTest(i=model):
                ModelConversionTest.convert_model_and_check_predictions(
                    MODELS_METADATA, model
                )

    @staticmethod
    def convert_model_and_check_predictions(data_path, model_name_versioned):
        # Compose the paths needed forward.
        model_dir_path = os.path.join(data_path, model_name_versioned)
        model_generated_code = os.path.join(RUST_GENERATED_CODE, model_name_versioned)
        model_generated_rust_code = os.path.join(
            model_generated_code, "rust_generated_code"
        )
        model_generated_code_expected = os.path.join(
            RUST_GENERATED_CODE_EXPECTED, model_name_versioned
        )

        # 1. Before running the test, make sure the generated_code path is removed
        if os.path.exists(RUST_GENERATED_CODE):
            shutil.rmtree(RUST_GENERATED_CODE)

        # 2. Run TensorFlow --> Rust conversion
        ModelConversionTest.run_tensorflow_to_rust_conversion(
            model_path=os.path.join(model_dir_path, "tf_model"),
            save_path=model_generated_code,
            fv_path=os.path.join(model_dir_path, "features.npz"),
        )
        print(model_generated_code)
        assert os.path.exists(model_generated_code)

        assert ModelConversionTest._are_dir_equal(
            model_generated_rust_code, model_generated_code_expected
        )

        # 3. Check that the Rust code compiles.
        ModelConversionTest.check_rust_code_compiles(
            code_path=model_generated_rust_code
        )

    @staticmethod
    def run_tensorflow_to_rust_conversion(
        model_path=None, save_path=None, fv_path=None
    ):
        """
        Run the command-line that converts TensorFlow model to Rust.

        python3 main.py \
        --path_to_tf_model tf_model/ \
        --path_to_save generated_code/ \
        --model_name HybridCNN \
        --path_to_fv features.npz
        """
        if not all((model_path, save_path, fv_path)):
            raise ValueError(
                "Expected to get model_path & save_path & fv_path as arguments!"
            )
        full_command = (
            "python3 main.py --path_to_tf_model {model_path} --path_to_save {save_path} "
            "--model_name Test --path_to_fv {fv_path}"
        )
        command = full_command.format(
            model_path=model_path, save_path=save_path, fv_path=fv_path
        )
        print(command)

        output = subprocess.getoutput(command)
        return output

    @staticmethod
    def check_rust_code_compiles(code_path=None):
        """
        Check that the Rust code compiles and check TensorFlow against Rust predictions.

        cargo fmt
        cargo test --release
        """
        # Make sure tests don't fail because of extra newlines/tabs/spaces.
        subprocess.run(["cargo", "fmt"], cwd=code_path)

        # Build crate and run tests.
        p = subprocess.run(
            ["cargo", "test", "--release"],
            cwd=code_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output_lines = p.stdout.decode("utf-8")
        assert output_lines

        # Go through the output and check that the number of "test result:" lines is equal to the
        # number of "test result: ok" lines.
        test_result_counts = 0
        test_result_ok_counts = 0
        for new_line in output_lines.split("\n"):
            if "test result:" in new_line:
                test_result_counts += 1
            if "test result: ok" in new_line:
                test_result_ok_counts += 1
            print(new_line)
        assert test_result_counts == test_result_ok_counts

    @staticmethod
    def _are_dir_equal(dir1, dir2):
        """
        Compare two directories recursively. Files in each directory are assumed to be equal if
        their names and contents are equal.

        Returns True if the directory trees are the same and there were no errors while accessing
                the directories or files, False otherwise.
        """
        dirs_cmp = filecmp.dircmp(dir1, dir2)
        if (
            len(dirs_cmp.left_only) > 0
            or len(dirs_cmp.right_only) > 0
            or len(dirs_cmp.funny_files) > 0
        ):
            return False

        (_, mismatch, errors) = filecmp.cmpfiles(
            dir1, dir2, dirs_cmp.common_files, shallow=False
        )

        if len(mismatch) > 0 or len(errors) > 0:
            return False

        for common_dir in dirs_cmp.common_dirs:
            new_dir1 = os.path.join(dir1, common_dir)
            new_dir2 = os.path.join(dir2, common_dir)
            if not ModelConversionTest._are_dir_equal(new_dir1, new_dir2):
                return False

        return True
