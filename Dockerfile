FROM python:3.7
#Set language parameters for the container
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
#Changing directory
WORKDIR /code
#Get all the requirements for PIP
COPY Code/requirements.txt .
#Install the dependencies
RUN pip install -r requirements.txt
#update the path
ENV PATH=/root/.local:$PATH
#Start the code (Use main for me)
CMD ["python","./main.py"]