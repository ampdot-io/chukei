module Resolver = Model_resolver.Make(Fs_store)

let () =
  print_endline "Chukei OCaml REPL";
  print_endline "Available commands:";
  print_endline "  models - List available models";
  print_endline "  resolve <model_id> - Resolve model configuration";
  print_endline "  quit - Exit REPL";
  print_endline "";
  
  let history_file = Filename.concat (Sys.getenv "HOME") ".chukei_history" in
  ignore (LNoise.history_load ~filename:history_file);
  ignore (LNoise.history_set ~max_length:100);
  
  let rec loop () =
    match LNoise.linenoise "chukei> " with
    | None -> print_endline "Goodbye!"
    | Some line ->
        ignore (LNoise.history_add line);
        ignore (LNoise.history_save ~filename:history_file);
        match String.split_on_char ' ' line with
        | ["models"] ->
            let models = Resolver.get_available_models () in
            List.iter (fun m ->
              Printf.printf "  %s (provider: %s, api_base: %s)\n" 
                m.Model_resolver.id m.Model_resolver.provider m.Model_resolver.api_base
            ) models;
            flush stdout;
            loop ()
        | ["resolve"; model_id] ->
            (match Resolver.get_model_config model_id with
            | Some config ->
                Printf.printf "Model: %s\n" model_id;
                Printf.printf "Provider: %s\n" 
                  (Option.value config.Model_resolver.provider ~default:"none");
                Printf.printf "API Base: %s\n" config.Model_resolver.api_base;
                Printf.printf "Headers: %s\n" 
                  (String.concat ", " (List.map (fun (k,v) -> k ^ ":" ^ v) config.Model_resolver.headers));
                flush stdout;
            | None ->
                Printf.printf "Model '%s' not found\n" model_id;
                flush stdout);
            loop ()
        | ["quit"] ->
            print_endline "Goodbye!"
        | [""] | [] ->
            loop ()
        | _ ->
            print_endline "Unknown command. Type 'quit' to exit.";
            loop ()
  in
  
  loop ()
