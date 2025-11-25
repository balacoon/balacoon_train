"""
Copyright 2022 Balacoon

Configuration storage based on omegaconf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Type, Union, cast
from importlib import import_module

from omegaconf import DictConfig, ListConfig, OmegaConf
from typing_extensions import Protocol


class IsDataClass(Protocol):
    """
    helper class to type the dataclasses
    """

    # https://stackoverflow.com/questions/54668000/type-hint-for-an-instance-of-a-non-specific-dataclass
    __dataclass_fields__: Dict


AttributeType = Union[str, int]  # the attribute is typically a string, but could be an index


class Config(object):
    """
    Config stores hyper-parameters for various blocks of modeling toolkit.
    It is based on Omegaconf package and is aimed to satisfy following functionality:

        - load/save from/to easily readable YAML file
        - fields of the config are easily accessible for both read/write, for ex. `config.generator.dimension`
        - it is easy to overwrite fields of the config from a command line,
          i.e. `config.parse("config.generator.dimension=256")`
        - config can set default values (even if those are absent) of hyperparameters from a dataclass
        - even if changes are done to sub-config, they are immediately accessible by all the references
          to config object. I.e. no copying is done.

    This allows to load/create config once and pass it between different blocks of the model.
    """

    def __init__(
        self,
        omega: Union[DictConfig, ListConfig] = None,
        attr: List[AttributeType] = [],
    ):
        """
        constructor of Config

        Parameters
        ----------
        omega: DictConfig
            omega config to wrap. is not copied in later access/manipulation, so should be set just once
        attr: List[str]
            list of attributes to access certain level of config
        """
        self._omega = omega
        self._attr = attr
        if self._omega is None:
            self._omega = OmegaConf.create()

    def _get_attr_omega(self) -> DictConfig:
        """
        Helper function that rewinds to the level of configuration
        specified by attributes stored in a wrapper.

        Returns
        -------
        subconf: DictConfig
            omega subconfiguration for internal use. for ex. for getting
            or setting values
        """
        # hint mypy that self._omega exists
        subconf = cast(DictConfig, self._omega)
        for attr in self._attr:
            subconf = subconf[attr]
        return subconf

    @staticmethod
    def load(yaml_path: str) -> Config:
        """
        Loads configuration from a YAML file

        Parameters
        ----------
        yaml_path: str
            path to a YAML to load

        Returns
        -------
        config: Config
            config instance
        """
        omega = OmegaConf.load(yaml_path)
        return Config(omega=omega)

    def save(self, yaml_path: str, store_subconfig: bool = False):
        """
        Saves the omega config to a path

        Parameters
        ----------
        yaml_path: str
            specifies where to save the config
        store_subconfig: bool
            whether to store parent configuration (typical case,
            since we want to keep complete config) or a sub-config,
            accessible via `self._attr`.
        """
        with open(yaml_path, "w") as fp:
            if store_subconfig:
                subconfig = self._get_attr_omega()
                OmegaConf.save(config=subconfig, f=fp.name)
            else:
                OmegaConf.save(config=self._omega, f=fp.name)

    def __getattr__(self, attr: AttributeType) -> Any:
        """
        forwards getting attribute from omega config,
        so fields of config are accessible via dot, as `config.generator.dimension`.
        no special treatment for `Config` members and functions is needed,
        because __getattr__ is a last resort of finding an object attribute
        returns either child config or actual value.
        """
        subconf = self._get_attr_omega()
        # few options how to handle requested subconf:
        # 1. if requested attribute doesn't exist - raise error
        # 2. if requested attribute is a value (any type including dict or list) - return it as is
        # 3. if requested attribute is a just another subconf - wrap it into Config, add extra attr and return
        if isinstance(attr, str) and attr not in subconf:
            raise AttributeError("There is no [{}]".format(attr))
        if isinstance(attr, int) and not OmegaConf.is_list(subconf):
            raise AttributeError("Can't access non-list config by index")
        if OmegaConf.is_dict(subconf[attr]) or OmegaConf.is_list(subconf[attr]):
            new_attr = list(self._attr)
            new_attr.append(attr)
            child = Config(omega=self._omega, attr=new_attr)
            return child
        else:
            return subconf[attr]

    def get(self, key: AttributeType, default_val: Any) -> Any:
        """
        Getter with a default value as in dict
        """
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default_val

    def __getitem__(self, key: AttributeType) -> Any:
        """
        just forward __getattr__, since handles
        both str and int
        """
        return self.__getattr__(key)

    def __setattr__(self, attr: AttributeType, val: Any):
        """
        if attributes are predefined for wrapper (omega or list of attributes),
        then set them in object, otherwise set attributes for omega config
        """
        if attr in ["_omega", "_attr"]:
            super().__setattr__(cast(str, attr), val)
        else:
            subconf = self._get_attr_omega()
            subconf[attr] = val

    def __setitem__(self, key: AttributeType, val: Any):
        """
        just forward __setattr__, since handles
        both str and int
        """
        self.__setattr__(key, val)

    def __contains__(self, attr: str):
        """
        checks if certain field is in a config, for ex. `dimension in config.model`
        """
        subconf = self._get_attr_omega()
        return hasattr(subconf, attr)

    def __str__(self) -> str:
        """
        returns string representation of the underlying config

        Returns
        -------
        conf_str: str
            readable string representation of configuration
        """
        subconf = self._get_attr_omega()
        return subconf.__str__()

    def __repr__(self):
        """
        returns string representation of the underlying config

        Returns
        -------
        conf_str: str
            unambiguous string representation of configuration
        """
        subconf = self._get_attr_omega()
        return subconf.__repr__()

    def __nonzero__(self) -> bool:
        """
        forwards omegaconf that allows to check
        if field is not none for example.

        Returns
        -------
        flag: bool
            True if config field is not none
        """
        subconf = self._get_attr_omega()
        return True if subconf else False

    def __len__(self) -> int:
        """
        returns lens of the configuration.
        shouldn't be called on atomic fields (int),
        but on config dicts/lists

        Returns
        -------
        len: int
            length of config list or dict
        """
        subconf = self._get_attr_omega()
        return len(subconf)

    def parse(self, dot_list: Union[str, List[str]]):
        """
        Parse dot-list (usually from command line) and overwrite/set
        those attributes in current config

        Parameters
        ----------
        dot_list: Union[str, List[str]]
            dot separated attribute and value to overwrite in current config.
            can be single string (for ex `"generator.dimension=512"`) or
            list of strings (for ex `["generator.dimension=512", "generator.num_layers=2"]`
        """
        # define the input type
        if isinstance(dot_list, str):
            dot_list = [dot_list]
        # prepend with current attributes
        prefix_dot_list = list(dot_list)
        assert all(
            [isinstance(x, str) for x in self._attr]
        ), "Can't prepend dot_list with integer attributes"
        for attr in self._attr[::-1]:
            prefix_dot_list = [str(attr) + "." + x for x in prefix_dot_list]
        # create a omega config from given dotlist
        new_omega = OmegaConf.from_dotlist(prefix_dot_list)
        # merge new omega with current omega, overwriting or setting values
        new_omega_dict = OmegaConf.to_container(new_omega, resolve=True)
        assert isinstance(new_omega_dict, dict), "config created from dot_list is not a dict!"
        for key, val in new_omega_dict.items():
            OmegaConf.update(
                cast(DictConfig, self._omega),
                str(key),
                val,
                force_add=True,
                merge=True,
            )

    def add_missing(self, data_class: Type[IsDataClass]):
        """
        Adds all missing values from a dataclass to a current config

        Parameters
        ----------
        data_class: IsDataClass
            data class which fields will be added to current config,
            if they are not already there
        """
        other_conf = OmegaConf.structured(data_class)
        self._merge(other_conf)

    def add_missing_config(self, other: Config):
        """
        Adds all missing values from another config to a current config.
        Can be used to add fields from a training config to an inference config.

        Parameters
        ----------
        other: Config
            other configuration to take values from
        """
        self._merge(cast(DictConfig, other._get_attr_omega()))

    def _merge(self, other_conf: Union[DictConfig, ListConfig]):
        """
        Internal function to add fields from `other_config` to a current
        one if they are not already there. This function is a backbone
        of `add_missing`, which is used to set default values from dataclasses
        or set config fields that were used in training.
        """
        subconf = self._get_attr_omega()
        # 1. in the config created from data class, update values that are present in current config
        other_conf = OmegaConf.merge(OmegaConf.to_container(other_conf, resolve=True), subconf)
        # 2. now set all the values from other_conf into current conf, if they are not there already
        other_conf_dict = OmegaConf.to_container(other_conf, resolve=True)
        assert isinstance(other_conf_dict, dict), "config created from data class is not a dict!"
        for key, val in other_conf_dict.items():
            OmegaConf.update(cast(DictConfig, subconf), str(key), val, force_add=True, merge=False)

    def to_dict(self) -> Dict[str, Any]:
        """
        Represent current configuration as a dict.
        Can be useful if configuration contains only parameters
        for the object constructor. Then object can be created as
        `SomeObject(**config.to_dict())`

        Returns
        -------
        conf_dict: Dict[str, Any]
            current config represented as dictionary
        """
        subconf = self._get_attr_omega()
        container = OmegaConf.to_container(subconf, resolve=True)
        return cast(Dict[str, Any], container)  # narrow down type, hinting mypy

    def to_list(self) -> List[Any]:
        """
        If underlying config is a list - converts it to primitive list.
        Executes same `to_container` underneath, but implemented as a separate
        method to narrow down the type
        """
        subconf = self._get_attr_omega()
        container = OmegaConf.to_container(subconf, resolve=True)
        return cast(List[Any], container)  # narrow down type, hinting mypy

    def resolve(self):
        """
        Forwards `resolve` method from OmegaConf.
        It resolves interpolation (cross-reference between config fields).
        Should be called once all custom changes (overwrite, data location)
        to a config are done.
        """
        OmegaConf.resolve(self._omega)

    def find_element_in_config_list(self, cls: str, name: str = None) -> int:
        """
        Specialized function for (sub)configs that are lists, to find an element
        of the list that corresponds to specified cls and (optionally) name.
        For example if config:

            config.processors = [{"cls": "Npz", "name": "melspec"}, {"cls": "Duration", "name": "durs"}]

        one can run:

            dur_processor_idx = config.processors.find_element_in_config_list("Duration", name="durs")

        to find specific processor

        Parameters
        ----------
        cls: str
            elements of list should have "cls" field (typically the case for configurable objects configs),
            which is compared to a provided one. Suffix of the `cls` can be provided so one don't need to
            provide package name for look up
        name: str
            optional field of list element. if provided, function checks if `name` is in list element and if
            it's value equals to provided one

        Returns
        -------
        idx: int
            index of the matching list element. If element not found, returns "-1"
        """
        subconf = self._get_attr_omega()
        assert OmegaConf.is_list(subconf), "Can't look up element, config is not a list"
        for i, element in enumerate(subconf):
            if not element:
                continue
            assert "cls" in element, "{} doesn't contain `cls` field, can't look up".format(
                str(element)
            )
            if element.cls.endswith(cls):
                if not name:
                    return i
                if "name" in element and element.name == name:
                    return i
        return -1


@dataclass
class ConfigurableConfig:
    """
    Base configuration that should be inherited by all the configs of configurable objects
    """

    cls: str = (
        "???"  # full class name to create it, usually `ClassName.__module__ + "." + ClassName.__name__`
    )


def _get_class(full_class_name: str) -> Callable:
    """
    helper function that given full name of the class returns a callable
    that upon execution will create an object.

    Parameters
    ----------
    full_class_name: str
        full name of the class (with package)

    Returns
    -------
    constructor: Callable
        constructor that upon execution creates object
    """
    try:
        module_path, class_name = full_class_name.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        raise ImportError("Failed to find {} specified in config".format(full_class_name))


def create_configurable(config: Config, **kwargs: Dict[str, Any]) -> Configurable:
    """
    Creates a configurable object out of config. Relies on `cls` field in config which
    specifies full name of the class, so it can be instantiated.
    """
    assert "cls" in config, "Can't create object from config, it is not `ConfigurableObjectConfig`"
    obj = _get_class(config.cls)(config, **kwargs)
    return obj
