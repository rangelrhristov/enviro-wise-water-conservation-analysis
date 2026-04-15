"""
Enviro Wise Water Conservation Analysis — Desktop GUI
======================================================
A customtkinter front-end for ``water_analysis.run_pipeline``.

Workflow
--------
1. Launch app. The CSV field is pre-populated with the bundled example
   AQUASTAT dataset so the user can run it with zero configuration.
2. User either accepts the default or picks their own CSV via the
   file dialog. Likewise for the output folder.
3. "Run Analysis" spins up a worker thread so the UI thread stays
   responsive; progress messages flow in via a queue.
4. When the run finishes, "Open Output Folder" becomes enabled and the
   user can jump straight to the generated charts.

Why a worker thread? The pipeline takes 10–20 seconds on first run
(pandas load, sklearn fit, six matplotlib renders). Running it on the
UI thread would freeze the window. A thread + queue is the simplest
pattern that keeps customtkinter happy.

Author: Rangel Hristov
"""

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

import water_analysis


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


def _resource_path(relative):
    """Find a bundled resource whether running from source or from a frozen exe."""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base = Path(__file__).parent
    return base / relative


class EnviroWiseApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Enviro Wise — Water Risk Analysis")
        self.geometry("960x720")
        self.minsize(820, 620)

        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._last_output: str | None = None

        self._build_ui()
        self._poll_log()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(22, 8))
        ctk.CTkLabel(
            header,
            text="Enviro Wise — Country Water-Risk Benchmarking",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="Destination Data Consultancy  •  FAO AQUASTAT data science pipeline",
            font=ctk.CTkFont(size=12),
            text_color="grey",
        ).pack(anchor="w", pady=(2, 0))

        form = ctk.CTkFrame(self)
        form.pack(fill="x", padx=24, pady=10)
        form.grid_columnconfigure(1, weight=1)

        # CSV row
        ctk.CTkLabel(form, text="CSV file:", width=110, anchor="w").grid(
            row=0, column=0, padx=(16, 6), pady=12, sticky="w"
        )
        default_csv = str(_resource_path("data/aquastat_water_data.csv"))
        self.csv_var = ctk.StringVar(value=default_csv)
        ctk.CTkEntry(form, textvariable=self.csv_var).grid(
            row=0, column=1, padx=6, pady=12, sticky="ew"
        )
        ctk.CTkButton(form, text="Browse...", width=110, command=self._pick_csv).grid(
            row=0, column=2, padx=(6, 16), pady=12
        )

        # Output row
        ctk.CTkLabel(form, text="Output folder:", width=110, anchor="w").grid(
            row=1, column=0, padx=(16, 6), pady=12, sticky="w"
        )
        self.out_var = ctk.StringVar(value=str(Path.cwd() / "outputs"))
        ctk.CTkEntry(form, textvariable=self.out_var).grid(
            row=1, column=1, padx=6, pady=12, sticky="ew"
        )
        ctk.CTkButton(form, text="Browse...", width=110, command=self._pick_output).grid(
            row=1, column=2, padx=(6, 16), pady=12
        )

        # Action bar
        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=24, pady=(4, 6))
        self.run_btn = ctk.CTkButton(
            actions,
            text="Run Analysis",
            width=170,
            height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._start_run,
        )
        self.run_btn.pack(side="left", padx=(0, 10))

        self.open_btn = ctk.CTkButton(
            actions,
            text="Open Output Folder",
            width=180,
            height=42,
            command=self._open_output_folder,
            state="disabled",
        )
        self.open_btn.pack(side="left")

        # Log area
        log_frame = ctk.CTkFrame(self)
        log_frame.pack(fill="both", expand=True, padx=24, pady=(10, 12))

        ctk.CTkLabel(
            log_frame,
            text="Analysis Log",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=16, pady=(12, 6))

        self.log_widget = ctk.CTkTextbox(log_frame, font=("Consolas", 11))
        self.log_widget.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.log_widget.insert("1.0", "Ready. Click 'Run Analysis' to begin.\n")
        self.log_widget.configure(state="disabled")

        # Status bar
        self.status_var = ctk.StringVar(value="Ready")
        ctk.CTkLabel(
            self,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=11),
            text_color="grey",
            anchor="w",
        ).pack(fill="x", padx=24, pady=(0, 10))

    # ------------------------------------------------------------------
    # Pickers
    # ------------------------------------------------------------------
    def _pick_csv(self):
        current = self.csv_var.get().strip()
        initial = Path(current).parent if current else Path.cwd()
        path = filedialog.askopenfilename(
            title="Select AQUASTAT CSV",
            initialdir=str(initial),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_var.set(path)

    def _pick_output(self):
        current = self.out_var.get().strip() or str(Path.cwd())
        path = filedialog.askdirectory(title="Select output folder", initialdir=current)
        if path:
            self.out_var.set(path)

    # ------------------------------------------------------------------
    # Log plumbing
    # ------------------------------------------------------------------
    def _log(self, msg):
        self._log_queue.put(str(msg))

    def _poll_log(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self.log_widget.configure(state="normal")
                self.log_widget.insert("end", msg + "\n")
                self.log_widget.see("end")
                self.log_widget.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    def _start_run(self):
        csv_path = self.csv_var.get().strip()
        out_dir = self.out_var.get().strip()

        if not csv_path or not Path(csv_path).exists():
            messagebox.showerror("Missing CSV", f"CSV file not found:\n{csv_path}")
            return
        if not out_dir:
            messagebox.showerror("Missing output", "Select an output folder.")
            return

        self.run_btn.configure(state="disabled", text="Running...")
        self.open_btn.configure(state="disabled")
        self.status_var.set("Running analysis...")
        self.log_widget.configure(state="normal")
        self.log_widget.delete("1.0", "end")
        self.log_widget.configure(state="disabled")

        worker = threading.Thread(
            target=self._pipeline_worker,
            args=(csv_path, out_dir),
            daemon=True,
        )
        worker.start()

    def _pipeline_worker(self, csv_path, out_dir):
        try:
            water_analysis.run_pipeline(csv_path, out_dir, log=self._log)
            self.after(0, lambda: self._on_success(out_dir))
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            err = f"{type(exc).__name__}: {exc}"
            self.after(0, lambda: self._on_failure(err))

    def _on_success(self, out_dir):
        self.run_btn.configure(state="normal", text="Run Analysis")
        self.open_btn.configure(state="normal")
        self.status_var.set(f"Done. Output: {out_dir}")
        self._last_output = out_dir

    def _on_failure(self, err):
        self.run_btn.configure(state="normal", text="Run Analysis")
        self.status_var.set("Error. See log.")
        self._log(f"\n[ERROR] {err}")
        messagebox.showerror("Analysis failed", err)

    # ------------------------------------------------------------------
    # Misc actions
    # ------------------------------------------------------------------
    def _open_output_folder(self):
        path = self._last_output or self.out_var.get().strip()
        if not path or not Path(path).exists():
            messagebox.showwarning("Not found", "Output folder does not exist.")
            return
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])


def main():
    app = EnviroWiseApp()
    app.mainloop()


if __name__ == "__main__":
    main()
