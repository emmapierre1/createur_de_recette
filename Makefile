# ----------------------------------
#          INSTALL & TEST
# ----------------------------------

# project id
PROJECT_ID=project-createur-recette



# bucket name
BUCKET_NAME = bucket-projet-createur-de-recette




install_requirements:
	@pip install -r requirements.txt



ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr createur_de_recette-*.dist-info
	@rm -fr createur_de_recette.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# project id - replace with your GCP project id
PROJECT_ID=le-wagon-779

# bucket name - replace with your GCP bucket name
BUCKET_NAME=wagon-data-779-createur_de_recette

# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west2

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

# path to the file to upload to GCP (the path to the file should be absolute or should match the directory where the make command is ran)
# replace with your local path to the `train_1k.csv` and make sure to put the path between quotes
LOCAL_PATH_RECIPES="createur_de_recette/data/recipes.csv"
LOCAL_PATH_INGREDIENTS="createur_de_recette/data/ingredients.csv"

# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
BUCKET_FOLDER=data

# name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

BUCKET_TRAINING_FOLDER = 'trainings'
PACKAGE_NAME=createur_de_recette
FILENAME=trainer
JOB_NAME=createur_de_recette_$(shell date +'%Y%m%d_%H%M%S')
PYTHON_VERSION=3.7
RUNTIME_VERSION=1.15

upload_data:
	@gsutil cp ${LOCAL_PATH_RECIPES} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}
	@gsutil cp ${LOCAL_PATH_INGREDIENTS} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier BASIC_GPU \
		--stream-logs


run_api:
	uvicorn api.fast:app --reload  # load web server with code autoreload
