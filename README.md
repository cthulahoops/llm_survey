## Running an experiment with your own prompt.

Fetch the models from the api.


```
llm_survey models fetch
```

Setup the prompt and export it as a yaml file:


```
llm_survey prompt new example
llm_survey prompt export --format yaml > example.yaml
```

Edit the prompt, and reimport it - no need to create the marking scheme yet.

```
llm_survey prompt import --format yaml < example.yaml
```

Run the experiment:

```
llm_survey run example
```

Build the site:

```
llm_survey build example
```

Start a http server and review the site:

```
cd out && python -m http.server && open http://localhost:8000
```
