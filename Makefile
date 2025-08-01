IMAGE_NAME=lyrics-app
CONTAINER_NAME=lyrics-container
PORT=8000

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -it -p 80:8000 --memory="512m" --cpus="1" --restart unless-stopped --name lyrics-container lyrics-app

stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

logs:
	docker logs -f $(CONTAINER_NAME)

clean:
	docker rmi $(IMAGE_NAME) || true