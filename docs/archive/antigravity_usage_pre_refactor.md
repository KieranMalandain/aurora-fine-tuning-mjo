# Antigravity Usage

## Purpose

This document explains how Antigravity should be used on this repository.

## Agent roles

### Coordinator
- reads docs and rules
- proposes plans
- reviews outputs from other agents
- avoids broad code edits at first

### Data / Evaluation agent
- owns data scripts and evaluation logic
- should not modify model internals without approval

### Model / Training agent
- owns model, losses, trainer, configs
- should not redefine evaluation protocols without approval

### Docs / Registry agent
- updates docs and experiment registry
- does not change scientific logic independently

## File ownership guidance

Avoid concurrent edits to the same file by multiple agents.

## First-task guidance

Before implementing, agents should:
- inspect repo tree
- read project docs
- summarize what is already implemented
- distinguish current code from target roadmap

## Preferred prompt style

Good:
- inspect these files
- propose minimal patch
- list exact files to modify
- do not touch unrelated modules

Bad:
- implement everything
- refactor broadly
- assume missing architecture details

## Human Notes