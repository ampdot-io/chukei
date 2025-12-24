module Resolver = Model_resolver.Make(Fs_store)

let () =
  let models_handler _ =
    let models = Resolver.get_available_models () in
    let json = 
      let model_list = List.map (fun m -> 
        `Assoc [("id", `String m.Model_resolver.id)]
      ) models in
      `Assoc [("data", `List model_list)]
    in
    let response_string = Yojson.Basic.to_string json in
    Dream.json response_string in

  let completions_handler request =
    let%lwt body = Dream.body request in
    let model_id = 
      try
        let json = Yojson.Basic.from_string body in
        let model_obj = Yojson.Basic.Util.member "model" json in
        Yojson.Basic.Util.to_string model_obj
      with
      | _ -> "" in
    
    match Resolver.get_model_config model_id with
    | Some _config ->
        let response_json = `Assoc [
          ("id", `String "cmpl-test");
          ("object", `String "text_completion");
          ("created", `Int (Unix.time () |> int_of_float));
          ("model", `String model_id);
          ("choices", `List [`Assoc [
            ("text", `String "Hello from OCaml!");
            ("index", `Int 0);
            ("logprobs", `Null);
            ("finish_reason", `String "stop");
          ]]);
          ("usage", `Assoc [
            ("prompt_tokens", `Int 10);
            ("completion_tokens", `Int 5);
            ("total_tokens", `Int 15);
          ]);
        ] in
        let response_string = Yojson.Basic.to_string response_json in
        Dream.json response_string
    | None ->
        let error_json = `Assoc [
          ("error", `Assoc [
            ("message", `String ("Could not find a provider that supports " ^ model_id));
            ("code", `Int 404);
          ]);
        ] in
        let response_string = Yojson.Basic.to_string error_json in
        Dream.json ~status:`Not_Found response_string in

  let chat_completions_handler request =
    let%lwt body = Dream.body request in
    let model_id = 
      try
        let json = Yojson.Basic.from_string body in
        let model_obj = Yojson.Basic.Util.member "model" json in
        Yojson.Basic.Util.to_string model_obj
      with
      | _ -> "" in
    
    match Resolver.get_model_config model_id with
    | Some _config ->
        let response_json = `Assoc [
          ("id", `String "chatcmpl-test");
          ("object", `String "chat.completion");
          ("created", `Int (Unix.time () |> int_of_float));
          ("model", `String model_id);
          ("choices", `List [`Assoc [
            ("index", `Int 0);
            ("message", `Assoc [
              ("role", `String "assistant");
              ("content", `String "Hello from OCaml chat!");
            ]);
            ("finish_reason", `String "stop");
          ]]);
          ("usage", `Assoc [
            ("prompt_tokens", `Int 10);
            ("completion_tokens", `Int 8);
            ("total_tokens", `Int 18);
          ]);
        ] in
        let response_string = Yojson.Basic.to_string response_json in
        Dream.json response_string
    | None ->
        let error_json = `Assoc [
          ("error", `Assoc [
            ("message", `String ("Could not find a provider that supports " ^ model_id));
            ("code", `Int 404);
          ]);
        ] in
        let response_string = Yojson.Basic.to_string error_json in
        Dream.json ~status:`Not_Found response_string in

  Dream.run ~port:6011
  @@ Dream.logger
  @@ Dream.router [
    Dream.get "/" (fun _ -> Dream.html "Chukei OCaml Model Server");
    Dream.get "/v1/models" models_handler;
    Dream.post "/v1/completions" completions_handler;
    Dream.post "/v1/chat/completions" chat_completions_handler;
  ]
