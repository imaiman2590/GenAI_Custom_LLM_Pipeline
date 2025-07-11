from prometheus_client import Counter, start_http_server

DOC_SUCCESS = Counter('document_extraction_success', 'Successful document extractions')
DOC_FAILURE = Counter('document_extraction_failure', 'Failed document extractions')

def start_metrics_server(port=8001):
    start_http_server(port)
