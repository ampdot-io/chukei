open Alcotest

module TestStore : Model_resolver.KV_STORE = struct
  let data = ref []
  
  let set key value = data := (key, value) :: !data
  let get key = List.assoc_opt key !data
  let list_keys () = List.map fst !data
  let clear () = data := []
end

module Resolver = Model_resolver.Make(TestStore)

let test_get_available_models_empty () =
  TestStore.clear ();
  let models = Resolver.get_available_models () in
  check int "no models when store is empty" 0 (List.length models)

let test_get_available_models_with_toml () =
  TestStore.clear ();
  TestStore.set "test-model.toml" {|
provider = "koboldcpp"
api_base = "http://localhost:5000"
|};
  let models = Resolver.get_available_models () in
  check int "one model when one toml file exists" 1 (List.length models);
  let model = List.hd models in
  check string "model id is correct" "test-model" model.Model_resolver.id;
  check string "provider is correct" "koboldcpp" model.Model_resolver.provider;
  check string "api_base is correct" "http://localhost:5000" model.Model_resolver.api_base

let test_get_available_models_with_subdirectory () =
  TestStore.clear ();
  TestStore.set "ggml-org/gemma-3-270m-GGUF.toml" {|
provider = "koboldcpp"
api_base = "http://127.0.0.1:55000"
|};
  let models = Resolver.get_available_models () in
  check int "one model when one toml file in subdir exists" 1 (List.length models);
  let model = List.hd models in
  check string "model id includes subdir" "ggml-org/gemma-3-270m-GGUF" model.Model_resolver.id

let test_config_toml_is_ignored () =
  TestStore.clear ();
  TestStore.set "config.toml" {|
[providers.kobold]
discovery_type = "koboldcpp"
|};
  let models = Resolver.get_available_models () in
  check int "config.toml should be ignored" 0 (List.length models)

let () =
  run "Model_resolver" [
    "get_available_models", [
      test_case "empty store" `Quick test_get_available_models_empty;
      test_case "with toml file" `Quick test_get_available_models_with_toml;
      test_case "with subdirectory" `Quick test_get_available_models_with_subdirectory;
      test_case "config.toml ignored" `Quick test_config_toml_is_ignored;
    ];
  ]
