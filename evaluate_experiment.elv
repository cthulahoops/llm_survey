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
  cohere/command-r
  nousresearch/nous-capybara-7b
  austism/chronos-hermes-13b
  01-ai/yi-large
  openai/gpt-3.5-turbo-0613
  google/gemma-2-9b-it
  huggingfaceh4/zephyr-orpo-141b-a35b
  anthropic/claude-2
  mistralai/mistral-large
  sophosympatheia/midnight-rose-70b
]

var evaluators = [
  anthropic/claude-3.5-sonnet
  openai/gpt-4-1106-preview
  openai/gpt-3.5-turbo-0125 
  anthropic/claude-3-haiku
]

each {|eval|
    llm_survey evaluate marshmallow -e $eval (each {|model| put -m $model} $sample_output_models)
} $evaluators
