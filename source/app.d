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

void main()
{
    auto p = 80; // number of training data nodes
    auto pt = 500; // number of fitting(evaluation) data nodes.

    auto sigma = 0.005; // noise amount in generated training data
    auto r = 0.3; // function radius
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
        .save("rbf_fit.png", 512, 512);
}
