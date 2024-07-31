from decimal import Decimal
from functools import cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path("templates")


@cache
def get_environment():
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape([])
    )

    return environment


@cache
def get_markdown():
    import markdown

    return markdown.Markdown(extensions=["markdown.extensions.fenced_code", "nl2br"])


def template_filter(name=None):
    def decorator(f):
        get_environment().filters[name or f.__name__] = f
        return f

    return decorator


OUTPUT_DIR = Path("out")


def render_to_file(template, output_file, **data):
    template = get_environment().get_template(template)
    rendered_html = template.render(**data)

    with open(OUTPUT_DIR / output_file, "w") as outfile:
        outfile.write(rendered_html)
    print(output_file)


@template_filter()
def model_name(model):
    return model.split("/")[-1]


@template_filter()
def model_company(model):
    return model.split("/")[-2]


@template_filter()
def model_file(model):
    name = model_name(model)
    return name[name.find("/") + 1 :].replace(":", "_").replace("/", "_") + ".html"


@template_filter()
def model_link(model):
    return f"<a href='{model_file(model)}'>{model_name(model)}</a>"


@template_filter()
def to_markdown(text):
    return get_markdown().convert(text)


@template_filter()
def cents(text):
    if text is None:
        return ""
    return "{:.2f}¢".format(Decimal(text) * 100)


@template_filter()
def average(items):
    return sum(items) / len(items)
