from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path("templates")

environment = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape([])
)


def template_filter(name=None):
    def decorator(f):
        environment.filters[name or f.__name__] = f
        return f

    return decorator


@template_filter()
def model_name(model):
    return model.split("/")[-1]


@template_filter()
def model_file(model):
    name = model_name(model)
    return name[name.find("/") + 1 :].replace(":", "_").replace("/", "_") + ".html"


@template_filter()
def model_link(model):
    return f"<a href='{model_file(model)}'>{model_name(model)}</a>"
