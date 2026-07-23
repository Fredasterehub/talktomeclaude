using System;
using System.IO;
using System.IO.Pipes;
using System.Threading;
public static class NativePipeBridge {
  public static int Main(string[] args) {
    if (args.Length != 1) return 2;
    try {
      using (var pipe = new NamedPipeClientStream(".", args[0], PipeDirection.InOut, PipeOptions.Asynchronous)) {
        pipe.Connect(15000);
        var stdin = Console.OpenStandardInput();
        var stdout = Console.OpenStandardOutput();
        Exception readError = null, writeError = null;
        var pipeToStdout = new Thread(() => {
          try { var b = new byte[4096]; int n; while ((n = pipe.Read(b, 0, b.Length)) > 0) { stdout.Write(b, 0, n); stdout.Flush(); } }
          catch (Exception e) { readError = e; }
        });
        var stdinToPipe = new Thread(() => {
          try { var b = new byte[4096]; int n; while ((n = stdin.Read(b, 0, b.Length)) > 0) { pipe.Write(b, 0, n); pipe.Flush(); } }
          catch (Exception e) { writeError = e; }
        });
        pipeToStdout.IsBackground = true; stdinToPipe.IsBackground = true;
        pipeToStdout.Start(); stdinToPipe.Start();
        stdinToPipe.Join();
        try { pipe.Close(); } catch { }
        pipeToStdout.Join(2000);
        if (writeError != null) throw writeError;
        if (readError != null && !(readError is ObjectDisposedException) && !(readError is IOException)) throw readError;
      }
      return 0;
    } catch (Exception e) { Console.Error.WriteLine(e); return 1; }
  }
}
