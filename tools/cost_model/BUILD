package(default_visibility = ["//visibility:public"])

licenses(["by_exception_only"])

load("@rules_python//python:defs.bzl", "py_test", "py_binary")
load("//bzl:plaidml.bzl", "plaidml_py_library", "plaidml_py_wheel")

py_test(
    name = "generator",
    timeout = "long",
    srcs = [
        "generator.py",
    ],
    tags = [
    ],
    deps = [
        "//plaidml/keras:keras",
        "//testing:plaidml_py",
    ],
)

