gcloud projects add-iam-policy-binding signtranslate-236821  \
    --member serviceAccount:$TPU_ACCOUNT --role roles/ml.serviceAgent
