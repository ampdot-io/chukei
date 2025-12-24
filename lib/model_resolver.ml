open Base

type model_info = {
  id : string;
  provider : string;
  api_base : string;
}

type model_config = {
  provider : string option;
  api_base : string;
  headers : (string * string) list;
  body : (string * string) list;
  kobold_path : string option;
  model_path : string option;
}

module type KV_STORE = sig
  val get : string -> string option
  val set : string -> string -> unit
  val list_keys : unit -> string list
  val clear : unit -> unit
end

module Make (Store : KV_STORE) = struct
  let parse_model_config id content =
    let lines = String.split_lines content in
    let find_value key =
      List.find_map lines ~f:(fun line ->
        let line = String.strip line in
        match String.lsplit2 line ~on:'=' with
        | Some (k, v) when String.equal (String.strip k) key ->
            let v = String.strip v in
            let v = String.strip ~drop:(Char.equal '"') v in
            Some v
        | _ -> None
      )
    in
    let provider = find_value "provider" in
    let api_base = find_value "api_base" in
    match api_base with
    | Some api_base -> Some { id; provider = Option.value provider ~default:"unknown"; api_base }
    | None -> None

  let get_available_models () : model_info list =
    let keys = Store.list_keys () in
    List.filter_map keys ~f:(fun key ->
      if String.is_suffix key ~suffix:".toml" && not (String.equal key "config.toml") then
        let id = String.chop_suffix_exn key ~suffix:".toml" in
        match Store.get key with
        | Some content -> parse_model_config id content
        | None -> None
      else
        None
    )

  let resolve_model model_id =
    let models = get_available_models () in
    List.find models ~f:(fun m -> String.equal m.id model_id)

  let is_model_available model_id =
    match resolve_model model_id with
    | Some _ -> true
    | None -> false

  let get_model_config model_id =
    match resolve_model model_id with
    | Some model -> Some {
        provider = Some model.provider;
        api_base = model.api_base;
        headers = [];
        body = [("model", model.id)];
        kobold_path = None;
        model_path = None;
      }
    | None -> None
end

module InMemoryStore : KV_STORE = struct
  let store : (string, string) Hashtbl.t = Hashtbl.create (module String)
  
  let set key value = Hashtbl.set store ~key ~data:value
  let get key = Hashtbl.find store key
  let list_keys () = Hashtbl.keys store
  let clear () = Hashtbl.clear store
end

module Default = Make(InMemoryStore)
