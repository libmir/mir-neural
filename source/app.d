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

import rbf;

alias Vector = Slice!(1, double*);
alias AesPointType = Aes!(Vector, "x", Vector, "y");
alias AesLineType = Aes!(Vector, "x", Vector, "y", string[], "colour");

void example_1d()
{
    /*
    Data fitting example.

    Train the RBF network with noisy data set from sine distribution,
    afterwards fit a generic linear space to trained model, and analyze results.
    */
    auto p = 80; // number of training data nodes
    auto pt = 500; // number of fitting(evaluation) data nodes.

    auto sigma = 0.005; // noise amount in generated training data
    auto r = 0.5; // function radius
    auto lambda = 1e-4; // regularization parameter

    // Create training data
    auto x = iota(p).map!(i => 1.0 - uniform(0.0, 1.0)).array.sliced(p);
    auto y = x.map!(v => sin(10.0 * v) + sigma * uniform(-double(p) / 2.0, cast(double)p) / 2.0).array.sliced(p);

    // Create query centers(xt), and create ideal fitting results(yt).
    auto xt = iota(1, pt + 1).map!(i => double(i) / double(pt)).array.sliced(pt);
    auto yt = xt.map!(v => sin(10.0 * v)).array.sliced(pt);

    // Create, setup, and train the network model with data defined above.
    auto network = RbfNetwork!(double, cauchy!double)()
        .lambda(lambda) // set regularization value
        .radius(r) // set function radius value
        .train(x, y); // train the network

    // fit query data to trained network.
    auto ft = network.fit(xt);

    // plot input data, perfect fit function (sine), and network fit values.
    GGPlotD()
        .put(geomPoint(AesPointType(x, y)))
        .put(geomLine(AesLineType(xt, yt, xt.map!(v => "green").array)))
        .put(geomLine(AesLineType(xt, ft, xt.map!(v => "blue").array)))
        .save("rbf_data_fit.png", 512, 512);
}

void example_nd() 
{
    /*
    Image value interpolation example.

    Demonstrates how 2D centers with vec3 values (RGB) can be used as training 
    data (e.g. color interpolation).
    */
    import dcv.core;
    import dcv.io;

    auto r = 5; // function radius
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
    auto network = RbfNetwork!(double, cauchy!double)()
        .lambda(lambda) // set regularization value
        .radius(r) // set function radius value
        .train(x, y); // train the network

    // fit query data to trained network.
    auto ft = network.fit(q.reshape(500*500, 2));

    // Reshape fitted data to match image size, and write it to disk.
    ft.reshape(500, 500, 3).asType!ubyte.imwrite(ImageFormat.IF_RGB, "rbf_color_interpolation.png");
}

void main(string[] args)
{
    example_1d();
    example_nd();
}

