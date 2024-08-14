FROM python:3.11.0

# create app directory
WORKDIR /home/cerdashukum/app

# set env variables
ENV PYTHONDONTWRITTERBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/home/cerdashukum/app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy project
COPY . .

#Expose the port your app runs on
EXPOSE 8080

# Run the server using uvicorn when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "15400", "--reload"]
