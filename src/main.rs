use bevy::{
    prelude::*,
    render::{
        mesh::{
            skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
            Indices, PrimitiveTopology,
        },
        render_asset::RenderAssetUsages,
    },
};

// 1. COMPONENTS
#[derive(Component)]
struct Player {
    speed: f32,
    velocity: Vec3,
    is_grounded: bool,
}

#[derive(Component)]
struct VirtualJoystick {
    direction: Vec2,
}

#[derive(Component, Clone, Copy)]
enum ActionButton {
    Triangle,
    Circle,
    Square,
    Cross,
}

// Custom component to identify bones for the idle animation
#[derive(Component)]
struct Bone(u16);

// Bone Indices
const BONE_PELVIS: u16 = 0;
const BONE_TORSO: u16 = 1;
const BONE_HEAD: u16 = 2;
const BONE_ARM_L_UPPER: u16 = 3;
const BONE_ARM_L_LOWER: u16 = 4;
const BONE_ARM_R_UPPER: u16 = 5;
const BONE_ARM_R_LOWER: u16 = 6;
const BONE_LEG_L_UPPER: u16 = 7;
const BONE_LEG_L_LOWER: u16 = 8;
const BONE_LEG_R_UPPER: u16 = 9;
const BONE_LEG_R_LOWER: u16 = 10;
const BONE_COUNT: usize = 11;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, (setup_scene, setup_ui))
        .add_systems(Update, (joystick_system, button_system, move_player, idle_animation_system))
        .run();
}

// ---------------------------------------------------------
// PROCEDURAL MESH BUILDER
// ---------------------------------------------------------
#[derive(Default)]
struct CharacterMeshBuilder {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    joint_indices: Vec<[u16; 4]>,
    joint_weights: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl CharacterMeshBuilder {
    fn add_face(
        &mut self,
        v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3,
        n: Vec3,
        b0: [u16; 4], w0: [f32; 4],
        b1: [u16; 4], w1: [f32; 4],
        b2: [u16; 4], w2: [f32; 4],
        b3: [u16; 4], w3: [f32; 4],
    ) {
        let i = self.positions.len() as u32;
        self.positions.extend([v0.to_array(), v1.to_array(), v2.to_array(), v3.to_array()]);
        self.normals.extend([n.to_array(); 4]);
        self.uvs.extend([[0.0, 0.0]; 4]);
        self.joint_indices.extend([b0, b1, b2, b3]);
        self.joint_weights.extend([w0, w1, w2, w3]);
        self.indices.extend([i, i + 1, i + 2, i + 2, i + 3, i]);
    }

    fn add_box(&mut self, center: Vec3, size: Vec3, bone: u16) {
        let hw = size.x / 2.0;
        let hh = size.y / 2.0;
        let hd = size.z / 2.0;
        let b = [bone, 0, 0, 0];
        let w = [1.0, 0.0, 0.0, 0.0];

        let p = [
            center + Vec3::new(-hw, -hh, hd),
            center + Vec3::new(hw, -hh, hd),
            center + Vec3::new(hw, hh, hd),
            center + Vec3::new(-hw, hh, hd),
            center + Vec3::new(-hw, -hh, -hd),
            center + Vec3::new(hw, -hh, -hd),
            center + Vec3::new(hw, hh, -hd),
            center + Vec3::new(-hw, hh, -hd),
        ];

        // Front, Back, Right, Left, Top, Bottom
        self.add_face(p[0], p[1], p[2], p[3], Vec3::Z, b, w, b, w, b, w, b, w);
        self.add_face(p[5], p[4], p[7], p[6], Vec3::NEG_Z, b, w, b, w, b, w, b, w);
        self.add_face(p[1], p[5], p[6], p[2], Vec3::X, b, w, b, w, b, w, b, w);
        self.add_face(p[4], p[0], p[3], p[7], Vec3::NEG_X, b, w, b, w, b, w, b, w);
        self.add_face(p[3], p[2], p[6], p[7], Vec3::Y, b, w, b, w, b, w, b, w);
        self.add_face(p[4], p[5], p[1], p[0], Vec3::NEG_Y, b, w, b, w, b, w, b, w);
    }

    fn add_limb(&mut self, top_pos: Vec3, joint_pos: Vec3, bottom_pos: Vec3, width: f32, depth: f32, top_bone: u16, bottom_bone: u16) {
        let hw = width / 2.0;
        let hd = depth / 2.0;
        
        let y_top = top_pos.y;
        let y_mid = joint_pos.y;
        let y_bot = bottom_pos.y;
        
        let cx = top_pos.x;
        let cz = top_pos.z;

        let b_top = [top_bone, 0, 0, 0];
        let w_top = [1.0, 0.0, 0.0, 0.0];
        let b_bot = [bottom_bone, 0, 0, 0];
        let w_bot = [1.0, 0.0, 0.0, 0.0];
        // 50/50 weighting at the joint
        let b_mid = [top_bone, bottom_bone, 0, 0];
        let w_mid = [0.5, 0.5, 0.0, 0.0];

        // Front face (Z+)
        self.add_face(
            Vec3::new(cx-hw, y_mid, cz+hd), Vec3::new(cx+hw, y_mid, cz+hd), Vec3::new(cx+hw, y_top, cz+hd), Vec3::new(cx-hw, y_top, cz+hd),
            Vec3::Z, b_mid, w_mid, b_mid, w_mid, b_top, w_top, b_top, w_top
        );
        self.add_face(
            Vec3::new(cx-hw, y_bot, cz+hd), Vec3::new(cx+hw, y_bot, cz+hd), Vec3::new(cx+hw, y_mid, cz+hd), Vec3::new(cx-hw, y_mid, cz+hd),
            Vec3::Z, b_bot, w_bot, b_bot, w_bot, b_mid, w_mid, b_mid, w_mid
        );
        
        // Back face (Z-)
        self.add_face(
            Vec3::new(cx+hw, y_mid, cz-hd), Vec3::new(cx-hw, y_mid, cz-hd), Vec3::new(cx-hw, y_top, cz-hd), Vec3::new(cx+hw, y_top, cz-hd),
            Vec3::NEG_Z, b_mid, w_mid, b_mid, w_mid, b_top, w_top, b_top, w_top
        );
        self.add_face(
            Vec3::new(cx+hw, y_bot, cz-hd), Vec3::new(cx-hw, y_bot, cz-hd), Vec3::new(cx-hw, y_mid, cz-hd), Vec3::new(cx+hw, y_mid, cz-hd),
            Vec3::NEG_Z, b_bot, w_bot, b_bot, w_bot, b_mid, w_mid, b_mid, w_mid
        );

        // Right face (X+)
        self.add_face(
            Vec3::new(cx+hw, y_mid, cz+hd), Vec3::new(cx+hw, y_mid, cz-hd), Vec3::new(cx+hw, y_top, cz-hd), Vec3::new(cx+hw, y_top, cz+hd),
            Vec3::X, b_mid, w_mid, b_mid, w_mid, b_top, w_top, b_top, w_top
        );
        self.add_face(
            Vec3::new(cx+hw, y_bot, cz+hd), Vec3::new(cx+hw, y_bot, cz-hd), Vec3::new(cx+hw, y_mid, cz-hd), Vec3::new(cx+hw, y_mid, cz+hd),
            Vec3::X, b_bot, w_bot, b_bot, w_bot, b_mid, w_mid, b_mid, w_mid
        );
        
        // Left face (X-)
        self.add_face(
            Vec3::new(cx-hw, y_mid, cz-hd), Vec3::new(cx-hw, y_mid, cz+hd), Vec3::new(cx-hw, y_top, cz+hd), Vec3::new(cx-hw, y_top, cz-hd),
            Vec3::NEG_X, b_mid, w_mid, b_mid, w_mid, b_top, w_top, b_top, w_top
        );
        self.add_face(
            Vec3::new(cx-hw, y_bot, cz-hd), Vec3::new(cx-hw, y_bot, cz+hd), Vec3::new(cx-hw, y_mid, cz+hd), Vec3::new(cx-hw, y_mid, cz-hd),
            Vec3::NEG_X, b_bot, w_bot, b_bot, w_bot, b_mid, w_mid, b_mid, w_mid
        );
        
        // Top cap
        self.add_face(
            Vec3::new(cx-hw, y_top, cz+hd), Vec3::new(cx+hw, y_top, cz+hd), Vec3::new(cx+hw, y_top, cz-hd), Vec3::new(cx-hw, y_top, cz-hd),
            Vec3::Y, b_top, w_top, b_top, w_top, b_top, w_top, b_top, w_top
        );
        // Bottom cap
        self.add_face(
            Vec3::new(cx-hw, y_bot, cz-hd), Vec3::new(cx+hw, y_bot, cz-hd), Vec3::new(cx+hw, y_bot, cz+hd), Vec3::new(cx-hw, y_bot, cz+hd),
            Vec3::NEG_Y, b_bot, w_bot, b_bot, w_bot, b_bot, w_bot, b_bot, w_bot
        );
    }
}

fn build_character_mesh() -> Mesh {
    let mut builder = CharacterMeshBuilder::default();

    // Torso
    builder.add_box(Vec3::new(0.0, 0.5, 0.0), Vec3::new(0.6, 1.0, 0.4), BONE_TORSO);
    // Head
    builder.add_box(Vec3::new(0.0, 1.2, 0.0), Vec3::new(0.4, 0.4, 0.4), BONE_HEAD);
    
    // Left Arm (with elbow loop cut)
    builder.add_limb(
        Vec3::new(-0.4, 0.9, 0.0), 
        Vec3::new(-0.4, 0.45, 0.0), 
        Vec3::new(-0.4, 0.0, 0.0), 
        0.2, 0.2, BONE_ARM_L_UPPER, BONE_ARM_L_LOWER
    );
    // Right Arm
    builder.add_limb(
        Vec3::new(0.4, 0.9, 0.0), 
        Vec3::new(0.4, 0.45, 0.0), 
        Vec3::new(0.4, 0.0, 0.0), 
        0.2, 0.2, BONE_ARM_R_UPPER, BONE_ARM_R_LOWER
    );
    // Left Leg (with knee loop cut)
    builder.add_limb(
        Vec3::new(-0.15, 0.0, 0.0), 
        Vec3::new(-0.15, -0.5, 0.0), 
        Vec3::new(-0.15, -1.0, 0.0), 
        0.2, 0.2, BONE_LEG_L_UPPER, BONE_LEG_L_LOWER
    );
    // Right Leg
    builder.add_limb(
        Vec3::new(0.15, 0.0, 0.0), 
        Vec3::new(0.15, -0.5, 0.0), 
        Vec3::new(0.15, -1.0, 0.0), 
        0.2, 0.2, BONE_LEG_R_UPPER, BONE_LEG_R_LOWER
    );

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, builder.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, builder.normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, builder.uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_JOINT_INDEX, bevy::render::mesh::VertexAttributeValues::Uint16x4(builder.joint_indices));
    mesh.insert_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, builder.joint_weights);
    mesh.insert_indices(Indices::U32(builder.indices));
    mesh
}

// ---------------------------------------------------------
// SETUP SYSTEMS
// ---------------------------------------------------------
fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut inverse_bindposes: ResMut<Assets<SkinnedMeshInverseBindposes>>,
) {
    // Ground Plane
    commands.spawn(PbrBundle {
        mesh: meshes.add(Cuboid::new(20.0, 0.2, 20.0)),
        material: materials.add(Color::srgb(0.3, 0.6, 0.3)),
        transform: Transform::from_xyz(0.0, -0.1, 0.0),
        ..default()
    });

    // Sun / Directional Light
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            shadows_enabled: true,
            illuminance: 10_000.0,
            ..default()
        },
        transform: Transform::from_xyz(8.0, 16.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    // Camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 4.0, 8.0).looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
        ..default()
    });

    // 1. Spawn the procedural mesh
    let char_mesh = build_character_mesh();

    // 2. Define the skeleton inverse bind poses.
    let mut bindposes = vec![Mat4::IDENTITY; BONE_COUNT];
    bindposes[BONE_PELVIS as usize] = Mat4::IDENTITY.inverse();
    bindposes[BONE_TORSO as usize] = Mat4::IDENTITY.inverse();
    bindposes[BONE_HEAD as usize] = Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)).inverse();
    bindposes[BONE_ARM_L_UPPER as usize] = Mat4::from_translation(Vec3::new(-0.4, 0.9, 0.0)).inverse();
    bindposes[BONE_ARM_L_LOWER as usize] = Mat4::from_translation(Vec3::new(-0.4, 0.45, 0.0)).inverse();
    bindposes[BONE_ARM_R_UPPER as usize] = Mat4::from_translation(Vec3::new(0.4, 0.9, 0.0)).inverse();
    bindposes[BONE_ARM_R_LOWER as usize] = Mat4::from_translation(Vec3::new(0.4, 0.45, 0.0)).inverse();
    bindposes[BONE_LEG_L_UPPER as usize] = Mat4::from_translation(Vec3::new(-0.15, 0.0, 0.0)).inverse();
    bindposes[BONE_LEG_L_LOWER as usize] = Mat4::from_translation(Vec3::new(-0.15, -0.5, 0.0)).inverse();
    bindposes[BONE_LEG_R_UPPER as usize] = Mat4::from_translation(Vec3::new(0.15, 0.0, 0.0)).inverse();
    bindposes[BONE_LEG_R_LOWER as usize] = Mat4::from_translation(Vec3::new(0.15, -0.5, 0.0)).inverse();

    let inv_bindposes_handle = inverse_bindposes.add(SkinnedMeshInverseBindposes::from(bindposes));

    // 3. Spawn the bone hierarchy
    let mut bones = vec![Entity::PLACEHOLDER; BONE_COUNT];
    
    let mut spawn_bone = |idx: u16, pos: Vec3| {
        let e = commands.spawn((SpatialBundle::from_transform(Transform::from_translation(pos)), Bone(idx))).id();
        bones[idx as usize] = e;
        e
    };

    let p_pelvis = spawn_bone(BONE_PELVIS, Vec3::ZERO);
    let p_torso = spawn_bone(BONE_TORSO, Vec3::ZERO);
    let p_head = spawn_bone(BONE_HEAD, Vec3::new(0.0, 1.0, 0.0));
    
    let p_arm_l_up = spawn_bone(BONE_ARM_L_UPPER, Vec3::new(-0.4, 0.9, 0.0));
    let p_arm_l_low = spawn_bone(BONE_ARM_L_LOWER, Vec3::new(0.0, -0.45, 0.0));
    
    let p_arm_r_up = spawn_bone(BONE_ARM_R_UPPER, Vec3::new(0.4, 0.9, 0.0));
    let p_arm_r_low = spawn_bone(BONE_ARM_R_LOWER, Vec3::new(0.0, -0.45, 0.0));
    
    let p_leg_l_up = spawn_bone(BONE_LEG_L_UPPER, Vec3::new(-0.15, 0.0, 0.0));
    let p_leg_l_low = spawn_bone(BONE_LEG_L_LOWER, Vec3::new(0.0, -0.5, 0.0));
    
    let p_leg_r_up = spawn_bone(BONE_LEG_R_UPPER, Vec3::new(0.15, 0.0, 0.0));
    let p_leg_r_low = spawn_bone(BONE_LEG_R_LOWER, Vec3::new(0.0, -0.5, 0.0));

    commands.entity(p_pelvis).push_children(&[p_torso, p_leg_l_up, p_leg_r_up]);
    commands.entity(p_torso).push_children(&[p_head, p_arm_l_up, p_arm_r_up]);
    commands.entity(p_arm_l_up).push_children(&[p_arm_l_low]);
    commands.entity(p_arm_r_up).push_children(&[p_arm_r_low]);
    commands.entity(p_leg_l_up).push_children(&[p_leg_l_low]);
    commands.entity(p_leg_r_up).push_children(&[p_leg_r_low]);

    // 4. Spawn the player mesh tied to the skeleton
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(char_mesh),
            material: materials.add(Color::srgb(0.2, 0.7, 0.9)),
            transform: Transform::from_xyz(0.0, 1.0, 0.0),
            ..default()
        },
        SkinnedMesh {
            inverse_bindposes: inv_bindposes_handle,
            joints: bones,
        },
        Player { 
            speed: 7.0,
            velocity: Vec3::ZERO,
            is_grounded: false,
        },
    )).push_children(&[p_pelvis]); // Attach skeleton root to player!
}

// ---------------------------------------------------------
// IDLE ANIMATION SYSTEM
// ---------------------------------------------------------
fn idle_animation_system(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &Bone)>,
) {
    let t = time.elapsed_seconds();
    
    for (mut transform, bone) in &mut query {
        match bone.0 {
            BONE_TORSO => {
                let breathe = 1.0 + (t * 2.0).sin() * 0.02;
                transform.scale = Vec3::new(1.0, breathe, 1.0);
            }
            BONE_ARM_L_UPPER => transform.rotation = Quat::from_rotation_x((t * 2.0).sin() * 0.1),
            BONE_ARM_R_UPPER => transform.rotation = Quat::from_rotation_x((t * 2.0).sin() * -0.1),
            BONE_ARM_L_LOWER | BONE_ARM_R_LOWER => transform.rotation = Quat::from_rotation_x(-0.2 + (t * 3.0).sin() * 0.05),
            BONE_LEG_L_UPPER => transform.rotation = Quat::from_rotation_x((t * 2.0).sin() * -0.05),
            BONE_LEG_R_UPPER => transform.rotation = Quat::from_rotation_x((t * 2.0).sin() * 0.05),
            BONE_LEG_L_LOWER | BONE_LEG_R_LOWER => transform.rotation = Quat::from_rotation_x(0.1 + (t * 3.0).cos() * 0.05),
            BONE_HEAD => transform.rotation = Quat::from_rotation_y((t * 1.5).sin() * 0.1),
            _ => {}
        }
    }
}

// ---------------------------------------------------------
// UI SETUP SYSTEM
// ---------------------------------------------------------
fn setup_ui(mut commands: Commands) {
    // Left Side: Joystick Base
    commands.spawn(NodeBundle {
        style: Style {
            width: Val::Px(150.0),
            height: Val::Px(150.0),
            position_type: PositionType::Absolute,
            left: Val::Px(25.0),
            bottom: Val::Px(25.0),
            ..default()
        },
        border_radius: BorderRadius::MAX,
        background_color: Color::srgba(0.5, 0.5, 0.5, 0.5).into(),
        ..default()
    });

    // Left Side: Joystick Knob (draggable)
    commands.spawn((
        NodeBundle {
            style: Style {
                width: Val::Px(60.0),
                height: Val::Px(60.0),
                position_type: PositionType::Absolute,
                left: Val::Px(70.0),   
                bottom: Val::Px(70.0), 
                ..default()
            },
            border_radius: BorderRadius::MAX,
            background_color: Color::srgba(0.8, 0.8, 0.8, 0.9).into(),
            ..default()
        },
        VirtualJoystick { direction: Vec2::ZERO },
    ));

    // Right Side: Container for PlayStation-style action buttons
    commands.spawn(NodeBundle {
        style: Style {
            width: Val::Px(200.0),
            height: Val::Px(200.0),
            position_type: PositionType::Absolute,
            right: Val::Px(50.0),
            bottom: Val::Px(50.0),
            ..default()
        },
        ..default()
    }).with_children(|parent| {
        let spawn_btn = |parent: &mut ChildBuilder, color: Color, label: &str, action: ActionButton, right: f32, bottom: f32| {
            parent.spawn((
                ButtonBundle {
                    style: Style {
                        width: Val::Px(60.0),
                        height: Val::Px(60.0),
                        position_type: PositionType::Absolute,
                        right: Val::Px(right),
                        bottom: Val::Px(bottom),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    border_radius: BorderRadius::MAX,
                    background_color: color.into(),
                    ..default()
                },
                action, 
            )).with_children(|btn| {
                btn.spawn(TextBundle::from_section(
                    label,
                    TextStyle {
                        font_size: 24.0,
                        color: Color::WHITE,
                        ..default()
                    },
                ));
            });
        };

        spawn_btn(parent, Color::srgb(0.0, 1.0, 0.0), "△", ActionButton::Triangle, 70.0, 140.0);
        spawn_btn(parent, Color::srgb(1.0, 0.0, 0.0), "◯", ActionButton::Circle, 0.0, 70.0);
        spawn_btn(parent, Color::srgb(0.0, 0.0, 1.0), "X", ActionButton::Cross, 70.0, 0.0);
        spawn_btn(parent, Color::srgb(1.0, 0.4, 0.7), "⬜", ActionButton::Square, 140.0, 70.0);
    });
}

// ---------------------------------------------------------
// INPUT & MOVEMENT SYSTEMS
// ---------------------------------------------------------
fn joystick_system(
    touches: Res<Touches>,
    windows: Query<&Window>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut joystick_query: Query<(&mut VirtualJoystick, &mut Style)>,
) {
    let Ok(window) = windows.get_single() else { return };
    let left_half = window.width() / 2.0;
    
    let mut pointer_pos = None;
    
    if let Some(touch) = touches.iter().next() {
        if touch.position().x < left_half {
            pointer_pos = Some(Vec2::new(touch.position().x, window.height() - touch.position().y));
        }
    } else if mouse_buttons.pressed(MouseButton::Left) {
        if let Some(cursor) = window.cursor_position() {
            if cursor.x < left_half {
                pointer_pos = Some(Vec2::new(cursor.x, window.height() - cursor.y));
            }
        }
    }
    
    for (mut joystick, mut style) in joystick_query.iter_mut() {
        let center = Vec2::new(100.0, 100.0); 
        if let Some(pos) = pointer_pos {
            let offset = pos - center;
            let max_dist = 75.0; 
            let dist = offset.length();
            
            if dist > 0.0 {
                let clamped_dist = dist.min(max_dist);
                let normalized = offset / dist;
                let new_pos = center + normalized * clamped_dist;
                
                style.left = Val::Px(new_pos.x - 30.0);
                style.bottom = Val::Px(new_pos.y - 30.0);
                
                joystick.direction = normalized * (clamped_dist / max_dist);
            }
        } else {
            style.left = Val::Px(center.x - 30.0);
            style.bottom = Val::Px(center.y - 30.0);
            joystick.direction = Vec2::ZERO;
        }
    }
}

fn button_system(
    interaction_query: Query<(&Interaction, &ActionButton), Changed<Interaction>>,
    mut player_query: Query<&mut Player>,
) {
    for (interaction, action) in &interaction_query {
        if *interaction == Interaction::Pressed {
            match action {
                ActionButton::Cross => {
                    println!("Jump / Accept");
                    if let Ok(mut player) = player_query.get_single_mut() {
                        if player.is_grounded {
                            player.velocity.y = 10.0;
                            player.is_grounded = false;
                        }
                    }
                }
                ActionButton::Square => println!("Attack! (Sword swing)"),
                ActionButton::Circle => println!("Dodge roll / Cancel!"),
                ActionButton::Triangle => println!("Open Inventory / Spell menu!"),
            }
        }
    }
}

fn move_player(
    time: Res<Time>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    joystick_query: Query<&VirtualJoystick>,
    mut query: Query<(&mut Player, &mut Transform)>,
) {
    let mut kb_direction = Vec3::ZERO;
    if keyboard_input.pressed(KeyCode::KeyW) || keyboard_input.pressed(KeyCode::ArrowUp) {
        kb_direction.z -= 1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyS) || keyboard_input.pressed(KeyCode::ArrowDown) {
        kb_direction.z += 1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyA) || keyboard_input.pressed(KeyCode::ArrowLeft) {
        kb_direction.x -= 1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyD) || keyboard_input.pressed(KeyCode::ArrowRight) {
        kb_direction.x += 1.0;
    }

    let mut js_direction = Vec3::ZERO;
    if let Ok(joystick) = joystick_query.get_single() {
        js_direction = Vec3::new(joystick.direction.x, 0.0, -joystick.direction.y);
    }
    
    let mut final_direction = kb_direction + js_direction;
    if final_direction.length_squared() > 1.0 {
        final_direction = final_direction.normalize();
    }
    
    let delta = time.delta_seconds();
    
    for (mut player, mut transform) in &mut query {
        transform.translation += final_direction * player.speed * delta;
        player.velocity.y -= 25.0 * delta; 
        transform.translation.y += player.velocity.y * delta;
        
        // Ground check for new mesh (feet at y=-1.0 relative to transform).
        // Since transform.y is the center, feet touch ground when transform.y = 1.0.
        if transform.translation.y <= 1.0 {
            transform.translation.y = 1.0; 
            player.velocity.y = 0.0;       
            player.is_grounded = true;     
        } else {
            player.is_grounded = false;    
        }
    }
}
