plugins {
    id("java")
}

group = "com.ed"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks{
    withType<JavaCompile>(){
        options.compilerArgs.add("--enable-preview")
        options.compilerArgs.add("--add-modules=jdk.incubator.vector")
        options.compilerArgs.add("-Xlint:preview")
    }

    withType<JavaExec>() {
        jvmArgs("--enable-preview")
        jvmArgs("--add-modules=jdk.incubator.vector")
    }
}

tasks.test {
    useJUnitPlatform()
}