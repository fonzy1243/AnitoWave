#include "engine/anito_wave.h"

int main(int argc, char *argv[]) {
    AnitoWave engine;

    engine.init();

    engine.run();

    engine.cleanup();

    return 0;
}