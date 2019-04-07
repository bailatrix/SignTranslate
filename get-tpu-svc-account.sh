curl -H "Authorization: Bearer $(gcloud auth print-access-token)"  \
    https://ml.googleapis.com/v1/projects/${PROJECT}:getConfig
