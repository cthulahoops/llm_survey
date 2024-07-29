#!/usr/bin/env elvish

var sample_output_models = [
  anthropic/claude-3-opus
  openai/gpt-4-turbo
  anthropic/claude-3.5-sonnet
  openai/gpt-3.5-turbo
  openai/gpt-4-1106-preview
  mistralai/mistral-large
  mancer/weaver
  jondurbin/bagel-34b
]

var evaluators = [
  anthropic/claude-3.5-sonnet
  openai/gpt-4-1106-preview
  openai/gpt-3.5-turbo-0125 
]

each {|eval|
    llm_survey evaluate marshmallow -e $eval (each {|model| put -m $model} $sample_output_models)
} $evaluators
