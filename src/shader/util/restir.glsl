#ifndef RESTIR_GLSL
#define RESTIR_GLSL

struct RestirReservoir {
    int pickedIdx;
    float sumW;
    float M;
    float pickedTargetPdf;
};

RestirReservoir restirEmptyReservoir() {
    RestirReservoir r;
    r.pickedIdx = -1;
    r.sumW = 0.0;
    r.M = 0.0;
    r.pickedTargetPdf = 0.0;
    return r;
}

bool restirIsValidReservoir(RestirReservoir r) {
    if (r.pickedIdx < 0) { return false; }
    if (isnan(r.sumW) || isinf(r.sumW) || r.sumW <= 0.0) { return false; }
    if (isnan(r.M) || isinf(r.M) || r.M <= 0.0) { return false; }
    if (isnan(r.pickedTargetPdf) || isinf(r.pickedTargetPdf) || r.pickedTargetPdf <= 1e-6) { return false; }
    return true;
}

RestirReservoir restirReservoirFromRaw(vec4 raw) {
    RestirReservoir r;
    r.pickedIdx = int(floor(raw.r + 0.5));
    r.sumW = raw.g;
    r.M = raw.b;
    r.pickedTargetPdf = raw.a;
    return restirIsValidReservoir(r) ? r : restirEmptyReservoir();
}

vec4 restirReservoirToRaw(RestirReservoir r) {
    return vec4(float(r.pickedIdx), r.sumW, r.M, r.pickedTargetPdf);
}

bool restirReservoirAddCandidate(inout RestirReservoir reservoir, int pickedIdx, float targetPdf, inout uint rngSeed) {
    if (targetPdf <= 1e-6) { return false; }

    reservoir.sumW += targetPdf;
    reservoir.M += 1.0;

    bool picked = rand(rngSeed) * reservoir.sumW < targetPdf;
    if (picked) {
        reservoir.pickedIdx = pickedIdx;
        reservoir.pickedTargetPdf = targetPdf;
    }
    return picked;
}

RestirReservoir restirCapReservoirM(RestirReservoir reservoir, float mMax) {
    if (!restirIsValidReservoir(reservoir)) { return restirEmptyReservoir(); }

    float cappedM = min(reservoir.M, mMax);
    float scale = cappedM / max(reservoir.M, 1.0);
    reservoir.M = cappedM;
    reservoir.sumW *= scale;
    return reservoir;
}

float restirTemporalReweight(RestirReservoir prevReservoir, float targetPdfCurrent, float ratioMax) {
    if (!restirIsValidReservoir(prevReservoir)) { return 0.0; }

    float ratio = clamp(targetPdfCurrent / max(prevReservoir.pickedTargetPdf, 1e-6), 0.0, ratioMax);
    return prevReservoir.sumW * ratio;
}

bool restirReservoirMergeCandidate(inout RestirReservoir reservoir, int pickedIdx, float pickedTargetPdf,
                                   float candidateWeight, float candidateM, inout uint rngSeed) {
    if (pickedIdx < 0 || pickedTargetPdf <= 1e-6 || candidateWeight <= 0.0 || candidateM <= 0.0) { return false; }

    reservoir.sumW += candidateWeight;
    reservoir.M += candidateM;

    bool picked = rand(rngSeed) * reservoir.sumW < candidateWeight;
    if (picked) {
        reservoir.pickedIdx = pickedIdx;
        reservoir.pickedTargetPdf = pickedTargetPdf;
    }
    return picked;
}

float restirUnbiasedContributionWeight(RestirReservoir reservoir) {
    if (!restirIsValidReservoir(reservoir)) { return 0.0; }
    return reservoir.sumW / max(reservoir.M * reservoir.pickedTargetPdf, 1e-6);
}

float restirJacobian(float targetPdfAtNeighbor, float targetPdfAtSelf, float jacobianMax) {
    return clamp(targetPdfAtNeighbor / max(targetPdfAtSelf, 1e-6), 0.0, jacobianMax);
}

#endif
