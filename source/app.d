import std.stdio;
import std.range;
import std.algorithm;
import std.random;
import std.math;

import mir.ndslice;
import mir.ndslice.iteration;
import mir.ndslice.algorithm;
import mir.glas;

import ggplotd.aes;
import ggplotd.geom;
import ggplotd.ggplotd;

import mir.glas.l3 : gemm;

import dcv.core;
import dcv.io;
import dcv.plot;

import mir.experimental.model.rbf;
import mir.experimental.ml.rbfann;


void modelExample()
{
    /*
    Demonstrates rbf model - low-level api.
    */
    alias rbf = gaussian;

    auto radius = 2.5; // function radius
    auto lambda = 1e-4; // regularization parameter

    // Training data
    auto x = slice!double(4, 2);
    auto y = slice!double(4, 3);

    // Query points:
    // Let's query values for each pixel in the image...
    auto i = indexSlice(500, 500).ndMap!(v => [double(v[0]), double(v[1])]).slice;
    auto q = slice!double(500*500, 2);

    foreach(e, id; lockstep(q.pack!1.byElement, i.byElement))
    {
        e[0] = id[0];
        e[1] = id[1];
    }

    // Coordinates of training node points.
    x[] = [[100, 100], [100, 400], [400, 100], [400, 400],];

    // Pixel (RGB) values assigned to corresponding nodes in training data set.
    y[] = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]];

    auto H = slice!double(x.length, x.length);
    auto Hq = slice!double(q.length, x.length);
    auto w = slice!double(y.shape);
    auto r = slice!double(q.length, y.length!1);

    // training
    designRbf!rbf(x, x, radius, H);
    ridgeGlobalWeights(y, H, lambda, w);

    // estimation (interpolation)
    designRbf!rbf(q, x, radius, Hq);
    gemm(1.0, Hq, w, 0.0, r);

    // show results
    r.reshape(500, 500, 3).ndMap!(v => cast(ubyte)v).slice.imshow();
    waitKey();
}

void networkExample_1d()
{
    /*
    Data fitting example.

    Train the RBF network with noisy data set from sine distribution,
    afterwards fit a generic linear space to trained model, and analyze results.
    */

    alias Vector = Slice!(1, double*);
    alias AesPointType = Aes!(Vector, "x", Vector, "y");
    alias AesLineType = Aes!(Vector, "x", Vector, "y", string[], "colour");

    auto p = 80; // number of training data nodes
    auto pt = 500; // number of fitting(evaluation) data nodes.

    auto sigma = 0.005; // noise amount in generated training data
    auto radius = 0.5; // function radius
    auto lambda = 1e-4; // regularization parameter

    // Create training data
    auto x = iota(p).map!(i => 1.0 - uniform(0.0, 1.0)).array.sliced(p);
    auto y = x.map!(v => sin(10.0 * v) + sigma * uniform(-double(p) / 2.0, cast(double)p) / 2.0).array.sliced(p);

    // Create query centers(xt), and create ideal fitting results(yt).
    auto xt = iota(1, pt + 1).map!(i => double(i) / double(pt)).array.sliced(pt);
    auto yt = xt.map!(v => sin(10.0 * v)).array.sliced(pt);

    // Create, setup, and train the network model with data defined above.
    auto network = RbfNetwork!(double, cauchy!double)(radius, lambda).train(x, y);

    // fit query data to trained network.
    auto ft = network.fit(xt);

    // plot input data, perfect fit function (sine), and network fit values.
    GGPlotD()
        .put(geomPoint(AesPointType(x, y)))
        .put(geomLine(AesLineType(xt, yt, xt.map!(v => "green").array)))
        .put(geomLine(AesLineType(xt, ft, xt.map!(v => "blue").array)))
        .save("rbf_data_fit.png", 500, 500);
}

void networkExample_nd() 
{
    /*
    Image value interpolation example.

    Demonstrates how 2D centers with vec3 values (RGB) can be used as training 
    data (e.g. color interpolation).
    */

    auto radius = 2.5; // function radius
    auto lambda = 1e-4; // regularization parameter

    // Training data
    auto x = slice!double(4, 2);
    auto y = slice!double(4, 3);

    // Query points
    auto q = slice!double(500, 500, 2);

    // Coordinates of training node points.
    x[] = [
        [100, 100],
        [100, 400],
        [400, 100],
        [400, 400],
    ];

    // Pixel (RGB) values assigned to corresponding nodes in training data set.
    y[] = [
        [255, 0, 0], 
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0]
    ];

    // Let's query values for each pixel in the image...
    foreach(i; 0 .. 500)
        foreach(j; 0 .. 500)
        {
            q[i, j, 0] = double(i);
            q[i, j, 1] = double(j);
        }

    // Create, setup, and train the network model with data defined above.
    auto network = RbfNetwork!(double, cauchy!double)(radius, lambda).train(x, y); // train the network

    // fit query data to trained network.
    auto ft = network.fit(q.reshape(500*500, 2));

    auto image = ft.reshape(500, 500, 3).ndMap!(v => cast(ubyte)v).slice.imwrite(ImageFormat.IF_RGB, "rbf_color_interpolation.png");
}
void main(string[] args)
{
    modelExample();
    networkExample_1d();
    networkExample_nd();
}
