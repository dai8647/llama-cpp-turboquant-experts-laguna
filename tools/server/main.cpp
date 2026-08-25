#include <cstdio>

int llama_server(int argc, char ** argv);

int main(int argc, char ** argv) {
    // unbuffered stderr so fatal messages survive abnormal termination
    setvbuf(stderr, NULL, _IONBF, 0);
    return llama_server(argc, argv);
}
