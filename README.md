# TrendMachine: Webpage Resilience

An interactive webpage resilience portal based on a mathematical model to calculate a normalized score to quantify the temporal resilience of a web page as a time-series data based on the historical observations of the page in web archives.

To run it locally (in Docker), clone this repository and build a Docker image:

```
$ docker image build -t resilience .
```

Run a container from the freshly built Docker image:

```
$ docker container run --rm -it -p 8501:8501 resilience
```

Access http://localhost:8501/ in a web browser for an interactive Resilience portal.
