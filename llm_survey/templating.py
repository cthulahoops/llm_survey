from pathlib import Path

import markdown
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path("templates")

environment = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape([])
)


markdown = markdown.Markdown(extensions=["markdown.extensions.fenced_code", "nl2br"])


def template_filter(name=None):
    def decorator(f):
        environment.filters[name or f.__name__] = f
        return f

    return decorator


def render_to_file(template, output_file, **data):
    template = environment.get_template(template)
    rendered_html = template.render(**data)

    with open(output_file, "w") as outfile:
        outfile.write(rendered_html)


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
    return markdown.convert(text)
