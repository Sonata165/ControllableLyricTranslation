# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import ArgumentParser, Namespace
from importlib import import_module

import numpy as np
from datasets import load_dataset
from packaging import version

import huggingface_hub

from .. import AutoConfig, AutoFeatureExtractor, AutoTokenizer, is_tf_available, is_torch_available
from ..utils import logging
from . import BaseTransformersCLICommand


if is_tf_available():
    import tensorflow as tf

    tf.config.experimental.enable_tensor_float_32_execution(False)

if is_torch_available():
    import torch


MAX_ERROR = 5e-5  # larger error tolerance than in our internal tests, to avoid flaky user-facing errors
TF_WEIGHTS_NAME = "tf_model.h5"


def convert_command_factory(args: Namespace):
    """
    Factory function used to convert a model PyTorch checkpoint in a TensorFlow 2 checkpoint.

    Returns: ServeCommand
    """
    return PTtoTFCommand(
        args.model_name, args.local_dir, args.new_weights, args.no_pr, args.push, args.extra_commit_description
    )


class PTtoTFCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        train_parser = parser.add_parser(
            "pt-to-tf",
            help=(
                "CLI tool to run convert a transformers model from a PyTorch checkpoint to a TensorFlow checkpoint."
                " Can also be used to validate existing weights without opening PRs, with --no-pr."
            ),
        )
        train_parser.add_argument(
            "--model-name",
            type=str,
            required=True,
            help="The model name, including owner/organization, as seen on the hub.",
        )
        train_parser.add_argument(
            "--local-dir",
            type=str,
            default="",
            help="Optional local directory of the model repository. Defaults to /tmp/{model_name}",
        )
        train_parser.add_argument(
            "--new-weights",
            action="store_true",
            help="Optional flag to create new TensorFlow weights, even if they already exist.",
        )
        train_parser.add_argument(
            "--no-pr", action="store_true", help="Optional flag to NOT open a PR with converted weights."
        )
        train_parser.add_argument(
            "--push",
            action="store_true",
            help="Optional flag to push the weights directly to `main` (requires permissions)",
        )
        train_parser.add_argument(
            "--extra-commit-description",
            type=str,
            default="",
            help="Optional additional commit description to use when opening a PR (e.g. to tag the owner).",
        )
        train_parser.set_defaults(func=convert_command_factory)

    @staticmethod
    def find_pt_tf_differences(pt_model, pt_input, tf_model, tf_input):
        """
        Compares the TensorFlow and PyTorch models, given their inputs, returning a dictionary with all tensor
        differences.
        """
        pt_outputs = pt_model(**pt_input, output_hidden_states=True)
        tf_outputs = tf_model(**tf_input, output_hidden_states=True)

        # 1. All output attributes must be the same
        pt_out_attrs = set(pt_outputs.keys())
        tf_out_attrs = set(tf_outputs.keys())
        if pt_out_attrs != tf_out_attrs:
            raise ValueError(
                f"The model outputs have different attributes, aborting. (Pytorch: {pt_out_attrs}, TensorFlow:"
                f" {tf_out_attrs})"
            )

        # 2. For each output attribute, computes the difference
        def _find_pt_tf_differences(pt_out, tf_out, differences, attr_name=""):

            # If the current attribute is a tensor, it is a leaf and we make the comparison. Otherwise, we will dig in
            # recursivelly, keeping the name of the attribute.
            if isinstance(pt_out, torch.Tensor):
                tensor_difference = np.max(np.abs(pt_out.detach().numpy() - tf_out.numpy()))
                differences[attr_name] = tensor_difference
            else:
                root_name = attr_name
                for i, pt_item in enumerate(pt_out):
                    # If it is a named attribute, we keep the name. Otherwise, just its index.
                    if isinstance(pt_item, str):
                        branch_name = root_name + pt_item
                        tf_item = tf_out[pt_item]
                        pt_item = pt_out[pt_item]
                    else:
                        branch_name = root_name + f"[{i}]"
                        tf_item = tf_out[i]
                    differences = _find_pt_tf_differences(pt_item, tf_item, differences, branch_name)

            return differences

        return _find_pt_tf_differences(pt_outputs, tf_outputs, {})

    def __init__(
        self,
        model_name: str,
        local_dir: str,
        new_weights: bool,
        no_pr: bool,
        push: bool,
        extra_commit_description: str,
        *args
    ):
        self._logger = logging.get_logger("transformers-cli/pt_to_tf")
        self._model_name = model_name
        self._local_dir = local_dir if local_dir else os.path.join("/tmp", model_name)
        self._new_weights = new_weights
        self._no_pr = no_pr
        self._push = push
        self._extra_commit_description = extra_commit_description

    def get_text_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained(self._local_dir)
        sample_text = ["Hi there!", "I am a batch with more than one row and different input lengths."]
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pt_input = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
        tf_input = tokenizer(sample_text, return_tensors="tf", padding=True, truncation=True)
        return pt_input, tf_input

    def get_audio_inputs(self):
        processor = AutoFeatureExtractor.from_pretrained(self._local_dir)
        num_samples = 2
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]
        raw_samples = [x["array"] for x in speech_samples]
        pt_input = processor(raw_samples, return_tensors="pt", padding=True)
        tf_input = processor(raw_samples, return_tensors="tf", padding=True)
        return pt_input, tf_input

    def get_image_inputs(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained(self._local_dir)
        num_samples = 2
        ds = load_dataset("cifar10", "plain_text", split="test")[:num_samples]["img"]
        pt_input = feature_extractor(images=ds, return_tensors="pt")
        tf_input = feature_extractor(images=ds, return_tensors="tf")
        return pt_input, tf_input

    def run(self):
        if version.parse(huggingface_hub.__version__) < version.parse("0.8.1"):
            raise ImportError(
                "The huggingface_hub version must be >= 0.8.1 to use this command. Please update your huggingface_hub"
                " installation."
            )
        else:
            from huggingface_hub import Repository, create_commit
            from huggingface_hub._commit_api import CommitOperationAdd

        # Fetch remote data
        repo = Repository(local_dir=self._local_dir, clone_from=self._model_name)

        # Load config and get the appropriate architecture -- the latter is needed to convert the head's weights
        config = AutoConfig.from_pretrained(self._local_dir)
        architectures = config.architectures
        if architectures is None:  # No architecture defined -- use auto classes
            pt_class = getattr(import_module("transformers"), "AutoModel")
            tf_class = getattr(import_module("transformers"), "TFAutoModel")
            self._logger.warn("No detected architecture, using AutoModel/TFAutoModel")
        else:  # Architecture defined -- use it
            if len(architectures) > 1:
                raise ValueError(f"More than one architecture was found, aborting. (architectures = {architectures})")
            self._logger.warn(f"Detected architecture: {architectures[0]}")
            pt_class = getattr(import_module("transformers"), architectures[0])
            try:
                tf_class = getattr(import_module("transformers"), "TF" + architectures[0])
            except AttributeError:
                raise AttributeError(f"The TensorFlow equivalent of {architectures[0]} doesn't exist in transformers.")

        # Load models and acquire a basic input for its modality.
        pt_model = pt_class.from_pretrained(self._local_dir)
        main_input_name = pt_model.main_input_name
        if main_input_name == "input_ids":
            pt_input, tf_input = self.get_text_inputs()
        elif main_input_name == "pixel_values":
            pt_input, tf_input = self.get_image_inputs()
        elif main_input_name == "input_features":
            pt_input, tf_input = self.get_audio_inputs()
        else:
            raise ValueError(f"Can't detect the model modality (`main_input_name` = {main_input_name})")
        tf_from_pt_model = tf_class.from_pretrained(self._local_dir, from_pt=True)

        # Extra input requirements, in addition to the input modality
        if config.is_encoder_decoder or (hasattr(pt_model, "encoder") and hasattr(pt_model, "decoder")):
            decoder_input_ids = np.asarray([[1], [1]], dtype=int) * pt_model.config.decoder_start_token_id
            pt_input.update({"decoder_input_ids": torch.tensor(decoder_input_ids)})
            tf_input.update({"decoder_input_ids": tf.convert_to_tensor(decoder_input_ids)})

        # Confirms that cross loading PT weights into TF worked.
        crossload_differences = self.find_pt_tf_differences(pt_model, pt_input, tf_from_pt_model, tf_input)
        max_crossload_diff = max(crossload_differences.values())
        if max_crossload_diff > MAX_ERROR:
            raise ValueError(
                "The cross-loaded TensorFlow model has different outputs, something went wrong! Exaustive list of"
                f" maximum tensor differences above the error threshold ({MAX_ERROR}):\n"
                + "\n".join(
                    [f"{key}: {value:.3e}" for key, value in crossload_differences.items() if value > MAX_ERROR]
                )
            )

        # Save the weights in a TF format (if needed) and confirms that the results are still good
        tf_weights_path = os.path.join(self._local_dir, TF_WEIGHTS_NAME)
        if not os.path.exists(tf_weights_path) or self._new_weights:
            tf_from_pt_model.save_weights(tf_weights_path)
        del tf_from_pt_model  # will no longer be used, and may have a large memory footprint
        tf_model = tf_class.from_pretrained(self._local_dir)
        conversion_differences = self.find_pt_tf_differences(pt_model, pt_input, tf_model, tf_input)
        max_conversion_diff = max(conversion_differences.values())
        if max_conversion_diff > MAX_ERROR:
            raise ValueError(
                "The converted TensorFlow model has different outputs, something went wrong! Exaustive list of maximum"
                f" tensor differences above the error threshold ({MAX_ERROR}):\n"
                + "\n".join(
                    [f"{key}: {value:.3e}" for key, value in conversion_differences.items() if value > MAX_ERROR]
                )
            )

        commit_message = "Update TF weights" if self._new_weights else "Add TF weights"
        if self._push:
            repo.git_add(auto_lfs_track=True)
            repo.git_commit(commit_message)
            repo.git_push(blocking=True)  # this prints a progress bar with the upload
            self._logger.warn(f"TF weights pushed into {self._model_name}")
        elif not self._no_pr:
            self._logger.warn("Uploading the weights into a new PR...")
            commit_descrition = (
                "Model converted by the [`transformers`' `pt_to_tf`"
                " CLI](https://github.com/huggingface/transformers/blob/main/src/transformers/commands/pt_to_tf.py)."
                "\n\nAll converted model outputs and hidden layers were validated against its Pytorch counterpart."
                f" Maximum crossload output difference={max_crossload_diff:.3e}; Maximum converted output"
                f" difference={max_conversion_diff:.3e}."
            )
            if self._extra_commit_description:
                commit_descrition += "\n\n" + self._extra_commit_description
            hub_pr_url = create_commit(
                repo_id=self._model_name,
                operations=[CommitOperationAdd(path_in_repo=TF_WEIGHTS_NAME, path_or_fileobj=tf_weights_path)],
                commit_message=commit_message,
                commit_description=commit_descrition,
                repo_type="model",
                create_pr=True,
            )
            self._logger.warn(f"PR open in {hub_pr_url}")
