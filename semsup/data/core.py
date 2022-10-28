import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import pytorch_lightning as pl
import gensim.downloader
from datasets import load_dataset, load_from_disk, ClassLabel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, IterableDataset, Dataset
from pytorch_lightning.trainer.supporters import CombinedLoader


class RepeatedDataset(Dataset):
    """ Dataset which wraps another dataset, repeating each entry `repetitions` number of times
    """
    def __init__(self, dataset:Dataset, repetitions:int):
        assert repetitions > 0
        self.dataset = dataset
        self.repetitions = repetitions

    def __len__(self):
        return len(self.dataset) * self.repetitions

    def __getitem__(self, index):
        index %= len(self.dataset)
        return self.dataset.__getitem__(index)


class LabelDataset(IterableDataset):
    """
    Iterable dataset that infinitely interates over randomly sampled labels.
    returns tokenized outputs
    Args:
        dataset: HF dataset with tokenized labels
        classlabels: HF datasets ClassLabel object. Will set the return order of the labels
        label_to_idx: dict mapping from class name to indices in the dataset
    """

    def __init__(self, dataset, classlabels, label_to_idx):
        super().__init__()
        self.label_to_idx = label_to_idx
        self.classlabels = classlabels
        self.dataset = dataset

    def __next__(self):
        fused = defaultdict(list)
        for i in range(self.classlabels.num_classes):
            label = self.classlabels.int2str(i)
            try:
                choice = int(np.random.choice(self.label_to_idx[label]))
            except:
                raise Exception(f"no choice for label: {label}")
            for k, v in self.dataset[choice].items():
                fused[k].append(v)
        return {k: torch.stack(v) for k, v in fused.items()}

    def __iter__(self):
        return self


@dataclass
class SemSupDataArgs:
    """Base argument dataclass for all SemSup datamodules. Inheriting classes should fill in train_classes and val_classes

    Args:
        label_tokenizer: name or path of Huggingface tokenizer
        train_label_json: path to the train label json file
        val_label_json: path to the val label json file. Set to train_label_json if val_label_json is None
        cache_dir: directory to cache the dataset
        train_classes: tuple of class names, defines the order in which the label descriptions are returned by the label dataloader
        val_classes: see train_classes. Will use train_classes as val_classes if it is None
        label_max_len: maximum length of the label tokenizer
        overwrite_label_cache (bool): force overwrite the cache file even if it exists
        setup_glove_embeddings (bool): add a 'glove_emb' feature to the dataset during setup
        setup_glove_for_json (bool): whether to construct a bov with a JSON input
        use_rand_embeddings (bool): when setting up "glove_emb" use random vectors instead. Ignored if setup_glove_embeddings is False
        val_repetitions (int): number of times to iterate over the validation set
        run_test (bool): run test instances for the dataset
    """

    label_tokenizer: str = None
    train_label_json: str = None
    val_label_json: str = None
    cache_dir: str = None
    train_classes: tuple = None
    val_classes: tuple = None
    label_max_len: int = 128
    # overwrite_label_cache: bool = False
    overwrite_label_cache: bool = True
    setup_glove_embeddings: bool = False
    setup_glove_for_json: bool = False
    use_rand_embeddings: bool = False
    val_repetitions: int = 1
    run_test: bool = False

    # these are automatically filled in for you in __post_init__()
    train_label_hash: str = None
    val_label_hash: str = None
    train_label_cache_path: str = None
    val_label_cache_path: str = None

    def __post_init__(self):
        assert Path(self.train_label_json).is_file()
        assert Path(self.cache_dir).is_dir()
        if self.val_label_json is not None:
            assert Path(self.val_label_json).is_file()
        else:
            self.val_label_json = self.train_label_json

        # make path formats consistent
        self.val_label_json = str(Path(self.val_label_json).absolute())
        self.train_label_json = str(Path(self.train_label_json).absolute())

        self.train_label_hash = self.hash_func(
            (self.label_tokenizer, self.train_label_json, self.label_max_len)
        )
        self.val_label_hash = self.hash_func(
            (self.label_tokenizer, self.val_label_json, self.label_max_len)
        )
        self.train_label_cache_path = str(
            Path(self.cache_dir).joinpath("lab_" + self.train_label_hash)
        )
        self.val_label_cache_path = str(
            Path(self.cache_dir).joinpath("lab_" + self.val_label_hash)
        )
        # print(f"\n\n\n\n\n\n{self.train_label_hash}\n\n\n\n\n\n\n")

    def hash_func(self, x):
        return hashlib.md5(json.dumps(x, sort_keys=True).encode("utf-8")).hexdigest()

    @property
    def hash(self):
        # use the entire data args to generate a unique hash value
        return self.hash_func(dataclasses.asdict(self))


class SemSupDataModule(pl.LightningDataModule):
    """Base Lightning DataModule for all SemSup tasks

    Args:
        args (SemSupDataArgs): general arguments for the data module
        batch_size (int): batch size used by Lightning Trainer
        val_batch_size (int or None): validation batch size, if different from batch_size
        num_workers (int): number of workers to spawn for the dataloader
    """

    def __init__(
        self,
        args: SemSupDataArgs,
        batch_size: int = 64,
        val_batch_size: int = None,
        num_workers: int = 0,
    ):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.num_workers = num_workers
        self.train_classlabel = ClassLabel(names=self.args.train_classes)
        self.val_classlabel = ClassLabel(names=self.args.val_classes)
        self.label_dataset = dict()  # label dataset will be made in self.setup()

        # just instantiating these somehow messes up the labels???? made into a local vars instead
        # if self.args.setup_glove_embeddings: # for glove vector setup
        #     self.glove_vectors = gensim.downloader.load("glove-wiki-gigaword-300")
        #     self.glove_vectors_keys = self.glove_vectors.key_to_index.keys()

        # handled by inheriting classes. OVERRIDE THESE
        self.dataset = {"train": None, "val": None, "test": None}

    def prepare_label_data(self):
        """Prepare the data. When subclassing make sure to call self.prepare_label_data() first in self.prepare_data()
        to prepare the label dataset
        """
        if (
            not Path(self.args.train_label_cache_path).exists()
            or self.args.overwrite_label_cache
        ):
            tokenizer = AutoTokenizer.from_pretrained(self.args.label_tokenizer)
            train_label_dataset = load_dataset(
                "json", data_files=self.args.train_label_json, split="train"
            )
            train_label_dataset = train_label_dataset.map(
                lambda x: tokenizer(
                    x["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.args.label_max_len,
                ),
                batched=True,
            )
            train_label_dataset.save_to_disk(self.args.train_label_cache_path)

        if (
            not Path(self.args.val_label_cache_path).exists()
            or self.args.overwrite_label_cache
        ):
            tokenizer = AutoTokenizer.from_pretrained(self.args.label_tokenizer)
            val_label_dataset = load_dataset(
                "json", data_files=self.args.val_label_json, split="train"
            )
            val_label_dataset = val_label_dataset.map(
                lambda x: tokenizer(
                    x["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.args.label_max_len,
                ),
                batched=True,
            )
            val_label_dataset.save_to_disk(self.args.val_label_cache_path)

    def _preprocess_json_for_glove(self, string):
        "Convert json-like string to a bag of words for GLOVE embeddings"
        def remove_from_str(s, to_remove):
            for r in to_remove:
                s = s.replace(r, "")
            return s

        words = []
        string = string.strip().replace("_", " ")
        string = remove_from_str(
            string,
            to_remove=":[],.;\n}{)("
        )
        for word in string.split():
            words.append(word.lower())
        return " ".join(list(set(words)))

    def _get_glove_embedding(self, sentence, glove_vectors):
        if self.args.use_rand_embeddings:
            return np.random.randn(300).astype(float)

        glove_vectors_keys = glove_vectors.key_to_index.keys()
        sum_of_embeddings = 0.0
        total = 0
        for word in sentence.strip().split():
            word = word.lower().replace(":", "").replace(",", "").replace(".", "").replace(";", "")
            if word in glove_vectors_keys:
                sum_of_embeddings += glove_vectors.vectors[
                    glove_vectors.get_index(word)
                ]
                total += 1
        
        # Average the embeddings
        if total == 0:
            # there are some empty JSONs, this handles those cases.
            return self._get_glove_embedding("none", glove_vectors)

        sum_of_embeddings /= total
        return np.array(sum_of_embeddings).astype(float)

    def _setup_label_dataset(self, dataset, classlabel):
        label_to_idx = defaultdict(list)
        dataset.map(lambda x, i: label_to_idx[x["label"]].append(i), with_indices=True)

        keep_columns = ["input_ids", "attention_mask"] # tokens to keep in set_format
        if "token_type_ids" in dataset.column_names:
            keep_columns += ["token_type_ids"]

        if self.args.setup_glove_embeddings:
            glove_vectors = gensim.downloader.load("glove-wiki-gigaword-300")
            if self.args.setup_glove_for_json:
                dataset = dataset.map(
                    lambda x: {"glove_emb": self._get_glove_embedding(
                        self._preprocess_json_for_glove(x["text"]), glove_vectors
                    )}
                )
            else:
                dataset = dataset.map(lambda x: {"glove_emb": self._get_glove_embedding(x["text"], glove_vectors)})
            dataset.set_format(type="torch", columns=keep_columns + ["glove_emb"])
        else:
            dataset.set_format(type="torch", columns=keep_columns)
        return LabelDataset(dataset, classlabel, label_to_idx)

    def setup_labels(self, stage=None):
        """Setup the data. When subclassing, call self.setup_labels() in self.setup() to setup the label dataset"""
        train_dataset = load_from_disk(self.args.train_label_cache_path)
        self.label_dataset["train"] = self._setup_label_dataset(
            train_dataset, self.train_classlabel
        )
        val_dataset = load_from_disk(self.args.val_label_cache_path)
        self.label_dataset["val"] = self._setup_label_dataset(
            val_dataset, self.val_classlabel
        )

    def train_dataloader(self):
        return CombinedLoader(
            {
                "input_loader": DataLoader(
                    self.dataset["train"],
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=True,
                ),
                "label_loader": DataLoader(
                    self.label_dataset["train"], num_workers=1
                ),
            },
            "min_size",
        )

    def val_dataloader(self):
        return CombinedLoader(
            {
                "input_loader": DataLoader(
                    RepeatedDataset(self.dataset["val"], repetitions=self.args.val_repetitions),
                    batch_size=self.val_batch_size,
                    num_workers=self.num_workers,
                ),
                "label_loader": DataLoader(
                    self.label_dataset["val"], num_workers=1
                ),
            },
            "min_size",
        )

    def test_dataloader(self):
        """ will not use this for now to avoid writing a separate
        test_step() logic in the modelcls
        """
        raise NotImplementedError