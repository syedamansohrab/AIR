# Java Backend Dockerfile
FROM maven:3.9-eclipse-temurin-17 AS build
WORKDIR /app
COPY backend-java/pom.xml .
COPY backend-java/src ./src
RUN mvn clean package -DskipTests

FROM eclipse-temurin:17-jre-jammy
WORKDIR /app
COPY --from=build /app/target/*.jar app.jar
RUN mkdir -p data-corpus lucene-index
EXPOSE 8080
CMD ["java", "-jar", "app.jar"]
